<template>
  <div class="translator-container">
    <div class="main-panel">
      <div class="left-container">
        <LanguageSelector
          v-model:game="selectedGameName"
          v-model:language="selectedLanguage"
          :enabled-games="enabledGames"
        />

        <div class="view-switcher" v-if="selectedFile">
          <button
            class="view-button"
            :class="{ active: viewMode === 'translated' }"
            @click="viewMode = 'translated'"
          >
            Перевод
          </button>
          <div class="divider"></div>
          <button
            class="view-button"
            :class="{ active: viewMode === 'original' }"
            @click="viewMode = 'original'"
          >
            Оригинал
          </button>
        </div>

        <ImageDropZone
          v-model:file="selectedFile"
          :view-mode="viewMode"
          @translation="handleTranslation"
          @error="handleImageError"
        />
      </div>

      <TextPanel
        :text="translationText"
        @copy="copyTranslation"
        @download="downloadImage"
        @delete="handleDeleteImage"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import LanguageSelector from '@/components/LanguageSelector.vue'
import ImageDropZone from '@/components/ImageDropZone.vue'
import TextPanel from '@/components/TextPanel.vue'
import langConfig from '@/assets/configs/lang_config.json'

import { useStore } from '@/stores/store'

const store = useStore()
const selectedGameName = ref('')
const selectedLanguage = ref('')
const selectedFile = ref(null)
const translationText = computed(() => store.translationResult)

const enabledGames = computed(() => {
  return Object.fromEntries(
    Object.entries(langConfig.games).filter(([_, config]) => config.enabled),
  )
})

const handleTranslation = (text) => {
  translationText.value = text
}

const downloadImage = () => {
  if (!selectedFile.value) return

  const link = document.createElement('a')
  link.download = 'translated.png'
  link.href = URL.createObjectURL(selectedFile.value)
  link.click()
  URL.revokeObjectURL(link.href)
}

const copyTranslation = () => {
  navigator.clipboard
    .writeText(translationText.value)
    .catch((err) => console.error('Ошибка копирования:', err))
}

const handleDeleteImage = () => {
  selectedFile.value = null
  store.imageFile = null
  store.resetDetectionData()
}

const viewMode = ref('translated')
const handleImageError = (error) => {
  console.error('Ошибка обработки изображения:', error)
}

onMounted(() => {
  const gameNames = Object.keys(enabledGames.value)
  if (gameNames.length > 0) {
    selectedGameName.value = gameNames[0]
  }
})
</script>

<style scoped>
.translator-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #060817;
}

.main-panel {
  display: flex;
  width: 90%;
  max-width: 1200px;
  height: 70vh;
  border: 4px solid #4f5565;
  border-radius: 15px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background-color: #3d4557;
}

.left-container {
  flex: 1;
  padding: 25px;
  display: flex;
  flex-direction: column;
  border-right: 4px solid #4f5565;
}

.view-switcher {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
  background: #323a47;
  border-radius: 30px;
  overflow: hidden;
  box-sizing: border-box;
}

.view-button {
  flex: 1;
  padding: 10px;
  background: none;
  cursor: pointer;
  font-weight: bold;
  font-size: 16px;
  color: #e8f0fe;
  border: none;
  border: 3px solid transparent;
  box-sizing: border-box;
  line-height: 1;
  font-family: 'GenshinFont';
}

.view-button.active {
  background: #3e4557;
  color: #baac97;
  border-radius: 30px;
  border: 3px solid #baac97;
}
</style>
