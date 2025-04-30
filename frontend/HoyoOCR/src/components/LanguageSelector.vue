<template>
  <div class="selection-panel">
    <div class="game-selector">
      <img :src="currentGameIcon" alt="Game icon" class="game-icon" @error="handleImageError" />
      <select v-model="game">
        <option v-for="(gameConfig, name) in enabledGames" :key="name" :value="name">
          {{ name }}
        </option>
      </select>
    </div>

    <div class="language-selector">
      <select v-model="language">
        <option v-for="lang in availableLanguages" :key="lang" :value="lang">
          {{ lang }}
        </option>
      </select>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import GI_icon from '@/assets/icons/GI_icon.png'
import HSR_icon from '@/assets/icons/HSR_icon.png'
import placeholderIcon from '@/assets/icons/placeholder-icon.png'

const props = defineProps({
  enabledGames: {
    type: Object,
    required: true,
  },
  game: {
    type: String,
    required: true,
  },
  language: {
    type: String,
    required: true,
  },
})

const emit = defineEmits(['update:game', 'update:language'])

const game = computed({
  get: () => props.game,
  set: (value) => emit('update:game', value),
})

const language = computed({
  get: () => props.language,
  set: (value) => emit('update:language', value),
})

const imageLoadError = ref(false)
const gameIcons = {
  'Genshin Impact': GI_icon,
  'Honkai: Star Rail': HSR_icon,
}

const availableLanguages = computed(() => {
  return props.enabledGames[props.game]?.languages || []
})

const currentGameIcon = computed(() => {
  if (imageLoadError.value || !props.game) {
    return placeholderIcon
  }
  return gameIcons[props.game] || placeholderIcon
})

const handleImageError = () => {
  imageLoadError.value = true
}

watch(game, (newVal) => {
  if (availableLanguages.value.length > 0) {
    language.value = availableLanguages.value[0]
  }
})
</script>

<style scoped>
.selection-panel {
  display: flex;
  gap: 15px;
  margin-bottom: 25px;
}

.game-selector,
.language-selector {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 12px;
}

.game-icon {
  width: 35px;
  height: 35px;
  object-fit: contain;
  border-radius: 50%;
}

select {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 16px;
  background-color: #f9f9f9;
  cursor: pointer;
}

select:focus {
  outline: none;
  border-color: #4a90e2;
}
</style>
