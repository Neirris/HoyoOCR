<template>
  <div class="right-container">
    <div class="panel-header">
      <button class="download-button" @click="handleDownload" v-if="text">Скачать</button>
      <button class="copy-button" @click="handleCopy" v-if="text">Копировать</button>
      <button class="delete-button" @click="handleDelete" v-if="text">
        <img src="@/assets/icons/CircleErrorRed.png" alt="Удалить" />
      </button>
    </div>

    <div class="translation-result" contenteditable="false">
      <p v-if="text">{{ text }}</p>
      <p v-else class="placeholder"></p>
    </div>
  </div>
</template>

<script setup>
import { toast } from 'vue3-toastify'
import 'vue3-toastify/dist/index.css'

defineProps({
  text: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['copy', 'download', 'delete'])

const handleCopy = () => {
  emit('copy')
  toast.info('Cкопировано в буфер обмена!', {
    position: 'top-center',
    autoClose: 1000,
  })
}

const handleDownload = () => {
  emit('download')
  toast.info('Изображение скачано!', {
    position: 'top-center',
    autoClose: 1000,
  })
}

const handleDelete = () => {
  emit('delete')
}
</script>

<style scoped>
.right-container {
  flex: 1;
  padding: 25px;
  display: flex;
  flex-direction: column;
  height: 100%;
  box-sizing: border-box;
}

.panel-header {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-bottom: 15px;
}

.copy-button,
.download-button {
  padding: 6px 14px;
  font-size: 16px;
  height: 36px;
  background-color: #e8f0fe;
  color: #060817;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s;
  font-family: 'GenshinFont';
}

.copy-button:hover,
.download-button:hover {
  background-color: #3a7bc8;
}

.delete-button {
  width: 36px;
  height: 36px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.delete-button img {
  width: 24px;
  height: 24px;
}

.translation-result {
  flex: 1;
  padding: 20px;
  border-radius: 8px;
  overflow-y: auto;
  background-color: #323a47;
  line-height: 1.6;
  box-sizing: border-box;
  color: #e8f0fe;
  font-size: 20px;
}

.placeholder {
  color: black;
  font-style: italic;
  text-align: center;
  margin-top: 40%;
  font-size: 30px;
}
</style>
