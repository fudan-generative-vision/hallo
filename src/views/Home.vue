<script setup lang="ts">
import WrappedSection from '@/components/WrappedSection.vue'
import sections from '@/index.json';

</script>

<template>
  <main>
    <WrappedSection v-for="(section, i) in sections" :key="i" :template="section.template" :props="section.props">
    </WrappedSection>
  </main>
</template>

<style lang="scss">
main {
  @apply w-full h-full flex flex-col items-center;

  >:nth-child(2n-1) {
    // @apply bg-white;
    // @apply dark:bg-black/50;
  }

  section {
    @apply w-full py-10 md:px-16 px-6;
    @apply flex flex-col justify-center items-center;
  }

  h1 {
    @apply text-7xl text-center;
  }

  h2 {
    @apply text-5xl text-center my-8;
  }

  h3 {
    @apply text-3xl text-center my-5;
  }
}
</style>
