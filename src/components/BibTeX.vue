<template>
    <section class="bibtex">
        <h3>BibTeX</h3>
        <pre class="bibtex-code"><code>{{ bibTeX }}</code></pre>
    </section>
</template>

<script setup lang="ts">
interface Props {
    bibTeX?: string
}
const { props } = defineProps<{ props: Props }>()
const bibTeX = props.bibTeX || ""
</script>

<style scoped lang="scss">
section {
    @apply w-full py-10 md:px-16 px-6;
    @apply flex flex-col justify-center items-center;
}

.bibtex-code {
    @apply border-gray-300 bg-gray-300/15 p-4 rounded-lg w-full overflow-auto;
    max-width: 960px;
}
</style>
