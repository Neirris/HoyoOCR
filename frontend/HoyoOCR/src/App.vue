<script setup>
import { RouterLink, RouterView } from 'vue-router'
import './assets/styles/css/main.css'
</script>

<template>
  <header>
    <div class="wrapper">
      <nav>
        <RouterLink to="/translate"></RouterLink>
      </nav>
    </div>
  </header>

  <RouterView />
</template>