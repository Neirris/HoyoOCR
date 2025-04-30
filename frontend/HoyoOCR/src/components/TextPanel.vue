<template>
  <div class="right-container">
    <div class="panel-header">
      <button class="download-button" @click="$emit('download')" v-if="text">Скачать</button>
      <button class="copy-button" @click="$emit('copy')" v-if="text">Копировать</button>
    </div>

    <div class="translation-result" contenteditable="false">
      <p v-if="text">{{ text }}</p>
      <p v-else class="placeholder">culmine</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  text: {
    type: String,
    default: '',
  },
})

defineEmits(['copy', 'download'])
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
  font-size: 14px;
  height: 36px;
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.copy-button:hover,
.download-button:hover {
  background-color: #3e8e41;
}

.translation-result {
  flex: 1;
  padding: 20px;
  border: 1px solid #eaeaea;
  border-radius: 8px;
  overflow-y: auto;
  background-color: #f9f9f9;
  line-height: 1.6;
  box-sizing: border-box;
  color: black;
  font-size: 20px;
}

.placeholder {
  /* color: #aaa; */
  color: black;
  font-style: italic;
  text-align: center;
  margin-top: 40%;
  font-size: 30px;
}
</style>
