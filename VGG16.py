import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

# ==========================================
# 1. 環境設定
# ==========================================

dataset_path = "Dataset_no_mask"
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # 若記憶體不足，請調小至 8 或 4

# 檢查路徑
if not os.path.exists(dataset_path):
    print(f"錯誤：找不到路徑 {dataset_path}")
    exit()

# ==========================================
# 2. 資料準備與類別權重計算
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

print("正在讀取訓練集...")
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="binary",
    subset="training",
    shuffle=True
)

print("正在讀取驗證集...")
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# --- 自動計算類別權重 (解決 Normal 為 0 的關鍵) ---
# 取得訓練集的類別標籤索引
train_labels = train_generator.classes
class_indices = train_generator.class_indices
print(f"類別對應: {class_indices}")

# 計算權重：總數 / (類別數 * 該類別數量)
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_vals))
print(f"已自動計算類別權重 (Class Weights): {class_weights}")

# ==========================================
# 3. 定義 Callbacks
# ==========================================
class SaveModelEvery5Epochs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            acc = logs.get("val_accuracy", 0)
            f1 = 0.0
            p = logs.get("val_precision", 0)
            r = logs.get("val_recall", 0)
            if (p + r) > 0: f1 = 2*p*r/(p+r)
            
            # 檔名標註階段
            stage = "FT" if self.model.optimizer.learning_rate.numpy() < 1e-4 else "Head"
            filename = f"vgg_{stage}_ep{epoch+1:03d}_acc{acc:.3f}_f1{f1:.3f}.h5"
            self.model.save(filename)
            print(f"\n[Checkpoint] 模型已儲存: {filename}")

# ==========================================
# 4. 階段一：凍結 VGG16，只訓練分類頭
# ==========================================
print("\n=== 進入階段一：訓練分類頭 (Frozen Base) ===")

base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(512, 512, 3)))

# 凍結所有底層
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4), # 初始學習率
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
)

# 設定階段一的 EarlyStopping
early_stopping_phase1 = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# 訓練階段一 (建議跑 20-30 epochs 讓分類頭穩定)
EPOCHS_PHASE_1 = 30
history_1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_1,
    callbacks=[early_stopping_phase1, SaveModelEvery5Epochs()],
    class_weight=class_weights # 套用權重
)

# ==========================================
# 5. 階段二：微調 (Fine-Tuning)
# ==========================================
print("\n=== 進入階段二：微調 VGG16 上層 (Unfreezing Blocks) ===")

# 1. 解凍 Base Model
base_model.trainable = True

# 2. 設定要微調的層數
# VGG16 Block 5 從 layer 15 開始, Block 4 從 layer 11 開始
# 我們這裡解凍 Block 5 和 Block 4 (更深層的特徵)
set_trainable = False
for layer in base_model.layers:
    if layer.name.startswith('block5') or layer.name.startswith('block4'):
        set_trainable = True
    else:
        set_trainable = False
    
    layer.trainable = set_trainable

# 3. 重新編譯模型 (關鍵：必須使用極小的學習率)
# 學習率設為 1e-5 (0.00001)，避免破壞預訓練權重
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
)

# 4. 繼續訓練
# 接續上一階段的 epoch 數
EPOCHS_PHASE_2 = 50 # 微調通常需要較多輪次
total_epochs = EPOCHS_PHASE_1 + EPOCHS_PHASE_2

early_stopping_phase2 = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)

history_2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=total_epochs,
    initial_epoch=history_1.epoch[-1] + 1, # 接續訓練
    callbacks=[early_stopping_phase2, SaveModelEvery5Epochs()],
    class_weight=class_weights # 繼續套用權重
)

# ==========================================
# 6. 整合歷史數據與繪圖
# ==========================================
# 合併兩個階段的 history
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
# 畫一條直線標示微調開始點
plt.axvline(x=len(history_1.history['accuracy']), color='green', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=len(history_1.history['accuracy']), color='green', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig("fine_tuning_process.png")
print("訓練圖表已儲存: fine_tuning_process.png")

# 儲存最終模型
final_model_name = "vgg16_finetuned_final.h5"
model.save(final_model_name)
print(f"最終微調模型已儲存: {final_model_name}")

# ==========================================
# 7. 最終評估
# ==========================================
print("\n正在進行最終評估...")
val_generator.reset()
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator, verbose=1)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

target_names = list(class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

print("\n最終分類報告：")
print(report)

with open("classification_report_finetuned.txt", "w", encoding="utf-8") as f:
    f.write(report)

# 混淆矩陣
try:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix (Fine-Tuned)")
    plt.savefig("confusion_matrix_finetuned.png")
    print("混淆矩陣已儲存")
except Exception as e:
    print(f"混淆矩陣錯誤: {e}")

