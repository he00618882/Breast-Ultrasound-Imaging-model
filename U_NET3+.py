import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, 
                          MaxPooling2D, UpSampling2D, Dropout, Dense, GlobalAveragePooling2D)
from keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from pathlib import Path

# ==========================================
# 1. 定義評估指標 (Dice & IoU)
# ==========================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

# ==========================================
# 2. 資料載入函數
# ==========================================
train_image_mask = {'images': [], 'masks': []}
test_image_mask = {'images': [], 'masks': []}

def load_image_mask(train_image_mask, test_image_mask, image_path, shape=512):
    """
    載入圖像和對應的遮罩，並分割成訓練與測試集。
    """
    file_names = os.listdir(image_path)
    # 過濾掉非圖片檔案，避免報錯
    file_names = [f for f in file_names if 'mask' not in f and f.endswith('.png')]
    
    # 處理括號檔名的邏輯
    partial_names = list(set(fn.split(')')[0] for fn in file_names))
    image_names = [name + ').png' for name in partial_names]
    mask_names = [name + ')_mask.png' for name in partial_names]
    
    train_test_split_index = int(len(image_names) * 0.8)
    
    for i, (image_name, mask_name) in enumerate(zip(image_names, mask_names)):
        img_path = os.path.join(image_path, image_name)
        mask_path = os.path.join(image_path, mask_name)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            try:
                image = cv2.resize(plt.imread(img_path), (shape, shape))
                mask = cv2.resize(plt.imread(mask_path), (shape, shape))
                
                # 確保 mask 是單通道
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                mask = np.expand_dims(mask, axis=-1) # (256, 256, 1)

                if i < train_test_split_index:
                    train_image_mask['images'].append(image)
                    train_image_mask['masks'].append(mask)
                else:
                    test_image_mask['images'].append(image)
                    test_image_mask['masks'].append(mask)
            except Exception as e:
                print(f"Error loading {image_name}: {e}")
    
    return train_image_mask, test_image_mask

# ==========================================
# 3. UNet 3+ 模型架構
# ==========================================
def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet_3plus(input_shape, initial_filters=32):
    """
    UNet 3+ Architecture with Full-Scale Skip Connections
    """
    cat_channels = initial_filters # 為了減少顯存消耗，拼接前的通道數統一
    
    # --- Encoder ---
    inputs = Input(input_shape)

    # Encoder 1
    e1 = conv_block(inputs, initial_filters)
    e1 = conv_block(e1, initial_filters)
    p1 = MaxPooling2D((2, 2))(e1)

    # Encoder 2
    e2 = conv_block(p1, initial_filters * 2)
    e2 = conv_block(e2, initial_filters * 2)
    p2 = MaxPooling2D((2, 2))(e2)

    # Encoder 3
    e3 = conv_block(p2, initial_filters * 4)
    e3 = conv_block(e3, initial_filters * 4)
    p3 = MaxPooling2D((2, 2))(e3)

    # Encoder 4
    e4 = conv_block(p3, initial_filters * 8)
    e4 = conv_block(e4, initial_filters * 8)
    p4 = MaxPooling2D((2, 2))(e4)

    # Encoder 5 (Bottleneck)
    e5 = conv_block(p4, initial_filters * 16)
    e5 = conv_block(e5, initial_filters * 16)

    # --- Decoder ---
    # 每個 Decoder 層接收來自：
    # 1. 同層級 Encoder (直接連接)
    # 2. 低層級 Encoder (透過 MaxPooling 下採樣)
    # 3. 高層級 Decoder (透過 UpSampling 上採樣)
    
    # Decoder 4
    # 來源: e1(down 8), e2(down 4), e3(down 2), e4(same), e5(up 2)
    d4_from_e1 = conv_block(MaxPooling2D((8, 8))(e1), cat_channels, kernel_size=3)
    d4_from_e2 = conv_block(MaxPooling2D((4, 4))(e2), cat_channels, kernel_size=3)
    d4_from_e3 = conv_block(MaxPooling2D((2, 2))(e3), cat_channels, kernel_size=3)
    d4_from_e4 = conv_block(e4, cat_channels, kernel_size=3)
    d4_from_e5 = conv_block(UpSampling2D((2, 2))(e5), cat_channels, kernel_size=3)
    
    d4 = concatenate([d4_from_e1, d4_from_e2, d4_from_e3, d4_from_e4, d4_from_e5])
    d4 = conv_block(d4, initial_filters * 16, kernel_size=3) # 聚合特徵

    # Decoder 3
    # 來源: e1(down 4), e2(down 2), e3(same), d4(up 2), e5(up 4)
    d3_from_e1 = conv_block(MaxPooling2D((4, 4))(e1), cat_channels, kernel_size=3)
    d3_from_e2 = conv_block(MaxPooling2D((2, 2))(e2), cat_channels, kernel_size=3)
    d3_from_e3 = conv_block(e3, cat_channels, kernel_size=3)
    d3_from_d4 = conv_block(UpSampling2D((2, 2))(d4), cat_channels, kernel_size=3)
    d3_from_e5 = conv_block(UpSampling2D((4, 4))(e5), cat_channels, kernel_size=3)

    d3 = concatenate([d3_from_e1, d3_from_e2, d3_from_e3, d3_from_d4, d3_from_e5])
    d3 = conv_block(d3, initial_filters * 8, kernel_size=3)

    # Decoder 2
    # 來源: e1(down 2), e2(same), d3(up 2), d4(up 4), e5(up 8)
    d2_from_e1 = conv_block(MaxPooling2D((2, 2))(e1), cat_channels, kernel_size=3)
    d2_from_e2 = conv_block(e2, cat_channels, kernel_size=3)
    d2_from_d3 = conv_block(UpSampling2D((2, 2))(d3), cat_channels, kernel_size=3)
    d2_from_d4 = conv_block(UpSampling2D((4, 4))(d4), cat_channels, kernel_size=3)
    d2_from_e5 = conv_block(UpSampling2D((8, 8))(e5), cat_channels, kernel_size=3)

    d2 = concatenate([d2_from_e1, d2_from_e2, d2_from_d3, d2_from_d4, d2_from_e5])
    d2 = conv_block(d2, initial_filters * 4, kernel_size=3)

    # Decoder 1
    # 來源: e1(same), d2(up 2), d3(up 4), d4(up 8), e5(up 16)
    d1_from_e1 = conv_block(e1, cat_channels, kernel_size=3)
    d1_from_d2 = conv_block(UpSampling2D((2, 2))(d2), cat_channels, kernel_size=3)
    d1_from_d3 = conv_block(UpSampling2D((4, 4))(d3), cat_channels, kernel_size=3)
    d1_from_d4 = conv_block(UpSampling2D((8, 8))(d4), cat_channels, kernel_size=3)
    d1_from_e5 = conv_block(UpSampling2D((16, 16))(e5), cat_channels, kernel_size=3)

    d1 = concatenate([d1_from_e1, d1_from_d2, d1_from_d3, d1_from_d4, d1_from_e5])
    d1 = conv_block(d1, initial_filters * 2, kernel_size=3)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ==========================================
# 4. 自定義 Callback：每5次儲存 + 檔名含指標
# ==========================================
class CustomSaver(Callback):
    def __init__(self, save_freq=5):
        super(CustomSaver, self).__init__()
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # 獲取指標，如果沒有則設為0
            acc = logs.get('accuracy', 0)
            iou = logs.get('iou_coef', 0)
            dice = logs.get('dice_coef', 0)
            val_loss = logs.get('val_loss', 0)
            
            # 格式化檔名
            file_name = f"unet3plus_ep{epoch+1:03d}_acc{acc:.4f}_iou{iou:.4f}_dice{dice:.4f}.h5"
            self.model.save(file_name)
            print(f"\nModel saved to {file_name}")

# ==========================================
# 5. 主程式與訓練流程
# ==========================================

# 超參數設定 (依需求修改)
initial_filters = 32  # 已更改為32
batch_size = 3        # 建議調小一點，因為UNet 3+參數量較大
epochs = 80 
learning_rate = 1e-4  # 建議初始學習率

# 準備訓練資料
categories = ['benign', 'normal'] 
base_dir = Path("Dataset_have_mask")

# 檢查路徑是否存在
if not base_dir.exists():
    print(f"警告：找不到資料集路徑 {base_dir}，請確認路徑正確。")
else:
    for category in categories:
        category_path = base_dir / category
        if category_path.exists():
            train_image_mask, test_image_mask = load_image_mask(
                train_image_mask=train_image_mask,
                test_image_mask=test_image_mask,
                image_path=category_path
            )

# 轉換為 Numpy Array
X_train = np.array(train_image_mask['images'])
y_train = np.array(train_image_mask['masks'])
X_test = np.array(test_image_mask['images'])
y_test = np.array(test_image_mask['masks'])

print(f"Training Data: {X_train.shape}, {y_train.shape}")
print(f"Testing Data: {X_test.shape}, {y_test.shape}")

if len(X_train) > 0:
    # 設置模型輸入和編譯
    segmentation_model = unet_3plus((512, 512, 3), initial_filters=initial_filters)
    
    segmentation_model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy', iou_coef, dice_coef] # 加入自定義指標
    )

    # 定義 Callbacks
    custom_saver = CustomSaver(save_freq=5) # 每5次儲存
    
    early_stopper = EarlyStopping(
        patience=15, 
        monitor='val_loss', 
        mode='min', 
        restore_best_weights=True, 
        verbose=1
    )

    # 訓練模型
    history = segmentation_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopper, custom_saver],
        verbose=1
    )

    # 保存訓練歷史的圖表
    history_df = pd.DataFrame(history.history)
    
    # 繪製 Accuracy
    plt.figure()
    history_df[['accuracy', 'val_accuracy']].plot(title='Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_plot.png')
    plt.clf()

    # 繪製 Loss
    plt.figure()
    history_df[['loss', 'val_loss']].plot(title='Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.clf()
    
    # 繪製 Dice & IoU
    plt.figure()
    history_df[['dice_coef', 'val_dice_coef']].plot(title='Dice Coefficient')
    plt.savefig('dice_plot.png')
    plt.clf()

    # 預測和存圖
    predictions = segmentation_model.predict(X_test)

    def save_image_mask_comparisons(test_images, predictions, test_masks, indices, output_dir='output_comparisons'):
        """
        保存原圖、預測遮罩與實際遮罩的比較。
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 確保 indices 不會超出測試集範圍
        valid_indices = [i for i in indices if i < len(test_images)]
        
        for i, idx in enumerate(valid_indices):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(test_images[idx])
            ax[0].set_title('Ultrasound Image')
            ax[0].axis('off')

            # 預測結果二值化 (Threshold 0.5)
            pred_mask = predictions[idx]
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            ax[1].imshow(pred_mask, cmap='gray')
            ax[1].set_title(f'Predicted Mask')
            ax[1].axis('off')

            ax[2].imshow(test_masks[idx], cmap='gray')
            ax[2].set_title('Actual Mask')
            ax[2].axis('off')

            plt.tight_layout()
            plt.savefig(f'{output_dir}/comparison_{idx}.png')
            plt.close()

    # 保存比較圖 (最多30張)
    indices_to_plot = list(range(30))
    save_image_mask_comparisons(X_test, predictions, y_test, indices_to_plot)

    print("訓練完成，結果與模型已儲存。")
else:
    print("錯誤：沒有載入任何訓練資料，請檢查資料集路徑。")