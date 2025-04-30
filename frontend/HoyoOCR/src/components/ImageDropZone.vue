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
      <div class="preview-wrapper">
        <img :src="previewUrl" class="preview-image" :class="{ 'error-blur': showError }" />
      </div>

      <div class="drag-overlay" v-show="isDragging">
        <div class="drop-message">
          <p>Перетащите изображение для замены</p>
        </div>
      </div>

      <div class="error-overlay" v-show="showError && !isDragging">
        <img src="@/assets/icons/CircleError.png" class="error-icon" />
        <p class="error-title">Текст не распознан</p>
        <p class="error-subtitle">Возможно, выбран неправильный язык</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  file: File,
  viewMode: {
    type: String,
    default: 'translated',
  },
})

const emit = defineEmits(['update:file', 'translation', 'error'])

const isDragging = ref(false)
const showError = ref(false)
const previewUrl = ref('')

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

  emit('update:file', file)
  showError.value = false

  setTimeout(() => {
    const shouldError = Math.random() > 0.7
    if (shouldError) {
      showError.value = true
      emit('error', 'Текст не распознан')
    } else {
      emit('translation', 'Текст распознан')
    }
  }, 1000)
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
  object-fit: contain;
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
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2;
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
  z-index: 3;
  pointer-events: none;
}

.drop-message {
  color: white;
  text-align: center;
  font-size: 24px;
  font-weight: 600;
  user-select: none;
}

.error-icon {
  width: 70px;
  height: 70px;
  margin-bottom: 20px;
}

.error-title {
  color: #ff4d4f;
  font-weight: 700;
  font-size: 22px;
  margin-bottom: 10px;
}

.error-subtitle {
  color: #ff4d4f;
  font-size: 16px;
  text-align: center;
}

.drop-content {
  text-align: center;
  padding: 20px;
}

.drop-content p {
  margin-bottom: 15px;
  color: #666;
  font-size: 16px;
}

.drop-content button {
  padding: 10px 20px;
  background-color: #4a90e2;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.drop-content button:hover {
  background-color: #3a7bc8;
}
</style>
