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
          :style="{ transform: `rotate(${rotationAngle}deg) scale(${scale})` }"
        />

        <div class="loading-overlay" v-if="store.isLoading">
          <div class="spinner"></div>
        </div>

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

      <div class="rotation-controls" v-if="(file && !isDragging) || showError">
        <button @click="zoomImage(-0.1)" class="zoom-button">
          <span>−</span>
        </button>
        <button @click="rotateImage(-15)" class="rotate-button">
          <img src="@/assets/icons/RotateLeft.png" alt="Повернуть влево" class="rotate-icon" />
        </button>
        <span class="rotation-angle">{{ rotationAngle }}° / {{ Math.round(scale * 100) }}%</span>
        <button @click="rotateImage(15)" class="rotate-button">
          <img src="@/assets/icons/RotateRight.png" alt="Повернуть вправо" class="rotate-icon" />
        </button>
        <button @click="zoomImage(0.1)" class="zoom-button">
          <span>+</span>
        </button>
      </div>

      <div class="drag-overlay" v-show="isDragging">
        <div class="drop-message">
          <p>Перетащите изображение<br />для замены</p>
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
const rotationAngle = ref(0)
const scale = ref(1)
const showError = computed(() => {
  return !store.isLoading && (store.error || !store.translationResult) && store.imageFile
})
const previewUrl = ref('')
const lastActionTime = ref(0)
const actionDebounce = ref(null)
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
        console.log('[watch:file] Loaded preview URL for file:', newFile.name)
      }
      reader.readAsDataURL(newFile)
    } else {
      if (previewUrl.value && previewUrl.value.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl.value)
      }
      previewUrl.value = ''
      rotationAngle.value = 0
      scale.value = 1
      console.log('[watch:file] Cleared file and reset transforms')
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
    emit('error', 'Неподдерживаемый формат файла')
    console.error('[processFile] Invalid file type:', file.type)
    return
  }

  console.log(
    '[processFile] Загружено изображение:',
    file.name,
    'Size:',
    file.size,
    'Type:',
    file.type,
  )
  rotationAngle.value = 0
  scale.value = 1
  store.imageFile = file
  store.resetDetectionData()
  emit('update:file', file)
  emit('translation', file)
}

const updateImageSize = () => {
  if (!imageEl.value) return
  imageSize.value = {
    width: imageEl.value.naturalWidth,
    height: imageEl.value.naturalHeight,
  }
  console.log('[updateImageSize] Image dimensions:', imageSize.value)
}

const getBoxStyle = (box) => {
  if (!imageEl.value || !imageWrapper.value) {
    console.warn('[getBoxStyle] No image or wrapper element')
    return {}
  }

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
    renderedWidth = wrapperWidth
    renderedHeight = wrapperWidth / imageAspect
  } else {
    renderedHeight = wrapperHeight
    renderedWidth = wrapperHeight * imageAspect
  }

  const scaleX = (renderedWidth / naturalWidth) * scale.value
  const scaleY = (renderedHeight / naturalHeight) * scale.value

  const offsetX = (wrapperWidth - renderedWidth * scale.value) / 2
  const offsetY = (wrapperHeight - renderedHeight * scale.value) / 2

  const style = {
    left: `${offsetX + box[0] * scaleX}px`,
    top: `${offsetY + box[1] * scaleY}px`,
    width: `${(box[2] - box[0]) * scaleX}px`,
    height: `${(box[3] - box[1]) * scaleY}px`,
    'border-radius': '4px',
  }

  console.log('[getBoxStyle] Box:', box, 'Scale:', scale.value, 'Style:', style)
  return style
}

const getSymbolForBox = (index) => {
  if (!store.translationResult || index >= store.translationResult.length) return ''
  return store.translationResult[index]
}

const createTransformedImage = async (angle, scaleFactor) => {
  if (!store.imageFile || !imageEl.value) {
    console.error('[createTransformedImage] No image file or element')
    return null
  }

  console.log(
    '[createTransformedImage] Processing image:',
    store.imageFile.name,
    'Angle:',
    angle,
    'Scale:',
    scaleFactor,
  )

  const img = imageEl.value
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  const width = img.naturalWidth
  const height = img.naturalHeight

  const rad = (angle * Math.PI) / 180
  const cos = Math.abs(Math.cos(rad))
  const sin = Math.abs(Math.sin(rad))
  const canvasWidth = Math.max(width, height * sin + width * cos)
  const canvasHeight = Math.max(height, width * sin + height * cos)

  canvas.width = canvasWidth * scaleFactor
  canvas.height = canvasHeight * scaleFactor

  console.log('[createTransformedImage] Canvas dimensions:', canvas.width, 'x', canvas.height)

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  ctx.save()

  ctx.translate(canvas.width / 2, canvas.height / 2)
  ctx.rotate(rad)
  ctx.scale(scaleFactor, scaleFactor)
  ctx.drawImage(img, -width / 2, -height / 2, width, height)

  ctx.restore()

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          console.error('[createTransformedImage] Failed to create blob')
          resolve(null)
          return
        }
        const fileName = store.imageFile.name.replace(/(\.[^.]+)$/, `_transformed$1`)
        const transformedFile = new File([blob], fileName, { type: store.imageFile.type })
        console.log(
          '[createTransformedImage] Created file:',
          fileName,
          'Size:',
          transformedFile.size,
          'Type:',
          transformedFile.type,
        )
        resolve(transformedFile)
      },
      store.imageFile.type,
      1.0,
    )
  })
}

const processTransformation = async () => {
  console.log('[processTransformation] Starting transformation:', {
    rotation: rotationAngle.value,
    scale: scale.value,
  })

  if (store.imageFile && imageEl.value) {
    const transformedFile = await createTransformedImage(rotationAngle.value, scale.value)
    if (transformedFile) {
      if (previewUrl.value && previewUrl.value.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl.value)
      }
      store.imageFile = transformedFile
      store.resetDetectionData()
      previewUrl.value = URL.createObjectURL(transformedFile)
      emit('update:file', transformedFile)
      emit('translation', transformedFile)
      console.log(
        '[processTransformation] Sent translation for file:',
        transformedFile.name,
        'Size:',
        transformedFile.size,
      )
    } else {
      emit('error', 'Ошибка при обработке изображения')
      console.error('[processTransformation] Transformation failed')
    }
  } else {
    emit('error', 'Нет изображения для обработки')
    console.error('[processTransformation] No image file or element')
  }
}

const rotateImage = (angle) => {
  const now = Date.now()
  if (now - lastActionTime.value < 300) {
    console.log('[rotateImage] Debounced: too soon')
    return
  }

  rotationAngle.value = (rotationAngle.value + angle) % 360
  lastActionTime.value = now

  if (actionDebounce.value) {
    clearTimeout(actionDebounce.value)
    console.log('[rotateImage] Cleared previous debounce')
  }

  actionDebounce.value = setTimeout(() => {
    processTransformation()
  }, 500)
}

const zoomImage = (delta) => {
  const now = Date.now()
  if (now - lastActionTime.value < 300) {
    console.log('[zoomImage] Debounced: too soon')
    return
  }

  const newScale = Math.max(0.1, scale.value + delta)
  scale.value = Number(newScale.toFixed(2))
  lastActionTime.value = now

  if (actionDebounce.value) {
    clearTimeout(actionDebounce.value)
    console.log('[zoomImage] Cleared previous debounce')
  }

  actionDebounce.value = setTimeout(() => {
    processTransformation()
  }, 500)
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
  border: 4px dashed #baac97;
  border-radius: 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  transition: all 0.3s;
  box-sizing: border-box;
  background-color: #3e4557;
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
  background-color: #323a47;
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
  max-width: 100%;
  max-height: 100%;
  position: relative;
  z-index: 1;
  transition: transform 0.3s ease;
}

.preview-image.error-blur {
  filter: blur(2px);
}

.rotation-controls {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  background-color: rgba(62, 69, 87, 0.8);
  padding: 8px 16px;
  border-radius: 24px;
  z-index: 5;
  backdrop-filter: blur(4px);
}

.rotate-button,
.zoom-button {
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  border-radius: 50%;
  transition: background-color 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.rotate-button {
  filter: invert();
}

.rotate-icon {
  width: 24px;
  height: 24px;
}

.zoom-button span {
  font-size: 24px;
  color: #e8f0fe;
  font-weight: bold;
}

.rotation-angle {
  color: #e8f0fe;
  font-size: 16px;
  min-width: 80px;
  text-align: center;
  font-family: 'GenshinFont';
}

.drag-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 10px;
  background-color: rgba(0, 0, 0, 0.8);
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
  background-color: rgba(38, 31, 38, 0.9);
}

.drop-message {
  color: #e8f0fe;
  text-align: center;
  font-size: 30px;
  user-select: none;
}

.error-icon {
  width: 70px;
  height: 70px;
}

.error-title {
  color: #ff4d4f;
  font-size: 30px;
  margin-bottom: 5px;
}

.error-subtitle {
  color: #ff4d4f;
  font-size: 20px;
  text-align: center;
  max-width: 100%;
  margin-top: 0;
}

.error-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: -20px;
}

.drop-content {
  text-align: center;
  font-weight: bold;
  padding: 20px;
}

.drop-content p {
  margin-bottom: 15px;
  color: #9197a3;
  font-size: 24px;
}

.empty-drop-zone.drag-active .drop-content p {
  color: #31302e;
}

.drop-content button {
  padding: 10px 20px;
  background-color: #cdad87;
  color: #31302e;
  border: none;
  font-weight: bold;
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
  color: #e8f0fe;
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
  background-color: rgba(153, 153, 153, 0.85);
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
