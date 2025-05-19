<template>
  <div
    class="drop-zone-container"
    @dragover.prevent="handleDragOver"
    @dragleave="handleDragLeave"
    @drop.prevent="handleDrop"
  >
    <div class="empty-drop-zone" :class="{ 'drag-active': isDragging && !file }" v-show="!file">
      <div class="drop-content">
        <p>Перетащите файл сюда или</p>
        <input
          type="file"
          id="file-input"
          accept=".png,.jpg,.jpeg,.webp"
          @change="handleFileSelect"
          hidden
        />
        <button @click="triggerFileInput">Выберите файл</button>
      </div>
    </div>

    <div class="image-preview" :class="{ empty: !file }" v-show="file">
      <div class="preview-wrapper" ref="imageWrapper">
        <img
          :src="previewUrl"
          class="preview-image"
          :class="{ 'error-blur': showError }"
          ref="imageEl"
          @load="updateImageSize"
        />

        <!-- Спиннер загрузки -->
        <div class="loading-overlay" v-if="store.isLoading">
          <div class="spinner"></div>
        </div>

        <!-- Символы поверх изображения -->
        <div
          class="symbols-overlay"
          v-if="viewMode === 'translated' && store.yoloDetection.boxes.length"
        >
          <div
            v-for="(box, index) in store.yoloDetection.boxes"
            :key="index"
            class="symbol-box"
            :style="getBoxStyle(box)"
          >
            <span class="symbol-char">{{ getSymbolForBox(index) }}</span>
          </div>
        </div>
      </div>

      <div class="drag-overlay" v-show="isDragging">
        <div class="drop-message">
          <p>Перетащите изображение для замены</p>
        </div>
      </div>

      <div class="error-overlay" v-show="showError && !isDragging">
        <img src="@/assets/icons/CircleErrorRed.png" class="error-icon" />
        <div class="error-text">
          <p class="error-title">Текст не распознан</p>
          <p class="error-subtitle">Возможно, выбран неправильный язык</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useStore } from '@/stores/store'

const props = defineProps({
  file: File,
  viewMode: {
    type: String,
    default: 'translated',
  },
})

const emit = defineEmits(['update:file', 'translation', 'error'])

const store = useStore()
const isDragging = ref(false)
const showError = computed(() => {
  return !store.isLoading && (store.error || !store.translationResult) && store.imageFile
})
const previewUrl = ref('')

const imageEl = ref(null)
const imageWrapper = ref(null)
const imageSize = ref({ width: 1, height: 1 })

watch(
  () => props.file,
  (newFile) => {
    if (newFile) {
      const reader = new FileReader()
      reader.onload = (e) => {
        previewUrl.value = e.target.result
      }
      reader.readAsDataURL(newFile)
    } else {
      previewUrl.value = ''
    }
  },
  { immediate: true },
)

const handleDragOver = (e) => {
  if (e.dataTransfer.types.includes('Files')) {
    isDragging.value = true
  }
}

const handleDragLeave = () => {
  isDragging.value = false
}

const triggerFileInput = () => {
  document.getElementById('file-input').click()
}

const handleFileSelect = (e) => {
  if (e.target.files.length) {
    processFile(e.target.files[0])
  }
  e.target.value = ''
}

const handleDrop = (e) => {
  isDragging.value = false
  const files = e.dataTransfer.files
  if (files.length) {
    processFile(files[0])
  }
}

const processFile = (file) => {
  const validTypes = ['image/png', 'image/jpeg', 'image/webp']
  if (!validTypes.includes(file.type)) {
    showError.value = true
    emit('error', 'Неподдерживаемый формат файла')
    return
  }

  console.log('[store] Загружено изображение:', file.name)
  emit('update:file', file)
  store.imageFile = file
  store.resetDetectionData() // Очищаем старые данные
  showError.value = false
}

// Обновить размер изображения
const updateImageSize = () => {
  if (!imageEl.value) return
  imageSize.value = {
    width: imageEl.value.naturalWidth,
    height: imageEl.value.naturalHeight,
  }
}

// Позиционирование боксов
const getBoxStyle = (box) => {
  if (!imageEl.value || !imageWrapper.value) return {}

  const img = imageEl.value
  const wrapper = imageWrapper.value

  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight
  const wrapperWidth = wrapper.clientWidth
  const wrapperHeight = wrapper.clientHeight

  const imageAspect = naturalWidth / naturalHeight
  const wrapperAspect = wrapperWidth / wrapperHeight

  let renderedWidth, renderedHeight

  if (imageAspect > wrapperAspect) {
    // Изображение ограничено по ширине
    renderedWidth = wrapperWidth
    renderedHeight = wrapperWidth / imageAspect
  } else {
    // Изображение ограничено по высоте
    renderedHeight = wrapperHeight
    renderedWidth = wrapperHeight * imageAspect
  }

  // Центрируем изображение внутри wrapper
  const offsetX = (wrapperWidth - renderedWidth) / 2
  const offsetY = (wrapperHeight - renderedHeight) / 2

  const scaleX = renderedWidth / naturalWidth
  const scaleY = renderedHeight / naturalHeight

  return {
    left: `${offsetX + box[0] * scaleX}px`,
    top: `${offsetY + box[1] * scaleY}px`,
    width: `${(box[2] - box[0]) * scaleX}px`,
    height: `${(box[3] - box[1]) * scaleY}px`,
    'border-radius': '4px',
  }
}

const getSymbolForBox = (index) => {
  if (!store.translationResult || index >= store.translationResult.length) return ''
  return store.translationResult[index]
}
</script>

<style scoped>
.drop-zone-container {
  position: relative;
  flex: 1;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

.empty-drop-zone {
  width: 100%;
  height: 100%;
  border: 2px dashed #ccc;
  display: flex;
  justify-content: center;
  align-items: center;
  transition: all 0.3s;
  box-sizing: border-box;
}

.empty-drop-zone.drag-active {
  border-color: #4a90e2;
  background-color: #e8f0fe;
}

.image-preview {
  width: 100%;
  height: 100%;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f9f9f9;
}

.image-preview.empty {
  border: 2px dashed #ccc;
}

.preview-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}

.preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain; /* Это гарантирует, что изображение будет масштабироваться, сохраняя пропорции */
  max-width: 100%; /* Ограничение максимальной ширины */
  max-height: 100%; /* Ограничение максимальной высоты */
  position: relative;
  z-index: 1;
}

.preview-image.error-blur {
  filter: blur(2px);
}

.drag-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 3;
  pointer-events: none;
}

.error-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 4;
  pointer-events: none;
  background-color: rgba(255, 255, 255, 0.9);
}

.drop-message {
  color: white;
  text-align: center;
  font-size: 30px;
  font-weight: 600;
  user-select: none;
}

.error-icon {
  width: 70px;
  height: 70px;
}

.error-title {
  color: #ff4d4f;
  font-weight: 700;
  font-size: 30px;
  margin-bottom: 5px; /* Уменьшим отступ */
}

.error-subtitle {
  color: #ff4d4f;
  font-size: 20px;
  text-align: center;
  max-width: 100%;
  margin-top: 0; /* Убираем верхний отступ */
}

.error-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: -20px; /* Поднимем текст выше */
}

.drop-content {
  text-align: center;
  padding: 20px;
}

.drop-content p {
  margin-bottom: 15px;
  color: #666;
  font-size: 24px;
}

.drop-content button {
  padding: 10px 20px;
  background-color: #4a90e2;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 20px;
  transition: background-color 0.3s;
}

.drop-content button:hover {
  background-color: #3a7bc8;
}

.symbols-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 2;
}

.symbol-box {
  position: absolute;
  background-color: rgba(0, 0, 0, 0.3);
  display: flex;
  justify-content: center;
  align-items: center;
  backdrop-filter: blur(4px);
}

.symbol-char {
  color: white;
  font-size: 60px;
  font-weight: bold;
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(200, 200, 200, 0.8);
  z-index: 3;
  display: flex;
  justify-content: center;
  align-items: center;
  pointer-events: none;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 5px solid #ccc;
  border-top-color: #4a90e2;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
