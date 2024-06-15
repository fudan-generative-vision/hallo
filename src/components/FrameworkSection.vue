<template>
    <section class="framework">
        <h3>Framework</h3>
        <div>
            <img :src="image">
            <br>
            <p>{{ description }}</p>
        </div>

    </section>
</template>

<script setup lang="ts">
interface Props {
    image?: string
    description?: string
}

const { props } = defineProps<{ props: Props }>()
const image = (props.image || "").startsWith("assets") ? new URL(`../${props.image}`, import.meta.url).href : props.image
const description = props.description
</script>

<style lang="scss" scoped>
.framework {
    * {
        max-width: 960px;
        @apply w-full;
    }

}
</style>
