// src/stores/store.js
import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import axios from 'axios'

export const useStore = defineStore('main', () => {
  const language = ref('')
  const imageFile = ref(null)
  const translationResult = ref('')
  const isLoading = ref(false)
  const error = ref(null)
  const yoloDetection = ref({
    boxes: [],
    classes: [],
    scores: [],
    imageUrl: null,
  })

  // Автоматический запрос на /api/translate при наличии и языка, и файла
  watch([language, imageFile], async ([lang, file]) => {
    if (!lang || !file) return

    console.log('[store] Язык выбран:', lang)
    console.log('[store] Файл получен:', file.name)
    console.log('[store] Запускается запрос на /api/translate...')

    const formData = new FormData()
    formData.append('file', file)

    isLoading.value = true
    error.value = null

    // Создаем превью изображения перед запросом
    const imageUrl = URL.createObjectURL(file)

    try {
      const response = await axios.post(
        `/api/translate?source_lang=${encodeURIComponent(lang)}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        },
      )

      console.log('[store] Получен ответ:', response.data)

      translationResult.value = response.data.text || ''

      // Обновляем только YOLO данные
      yoloDetection.value = {
        boxes: response.data.yolo_boxes || [],
        classes: response.data.yolo_classes || [],
        scores: response.data.yolo_scores || [],
        imageUrl: imageUrl,
      }
    } catch (err) {
      console.error('[store] Ошибка при переводе:', err)
      error.value = err.response?.data?.detail || err.message
      translationResult.value = '' // Вместо '[ошибка перевода]' пустая строка

      // При ошибке сохраняем только превью изображения
      yoloDetection.value = {
        boxes: [],
        classes: [],
        scores: [],
        imageUrl: imageUrl,
      }

      if (err.response) {
        console.error('Детали ошибки:', {
          status: err.response.status,
          data: err.response.data,
          headers: err.response.headers,
        })
      }
    } finally {
      isLoading.value = false
    }
  })

  // Функция для очистки превью изображения
  const clearDetectionData = () => {
    if (yoloDetection.value.imageUrl) {
      URL.revokeObjectURL(yoloDetection.value.imageUrl)
    }
    yoloDetection.value = {
      boxes: [],
      classes: [],
      scores: [],
      imageUrl: null,
    }
  }

  const resetDetectionData = () => {
    yoloDetection.value = {
      boxes: [],
      classes: [],
      scores: [],
      imageUrl: null,
    }
    translationResult.value = ''
  }

  return {
    language,
    imageFile,
    translationResult,
    isLoading,
    error,
    yoloDetection,
    clearError: () => {
      error.value = null
    },
    clearDetectionData,
    resetDetectionData,
  }
})
