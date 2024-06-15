<template>
    <section class="title">
        <h1>{{ title }}</h1>
        <h3>{{ subtitle }}</h3>

        <div v-for="(authorsItem, i) in authors" :key="i" class="authors">
            <div class="authors" :class="{ thin: i === authorsItem.length - 2 }">
                <span v-for="author, i in authorsItem" :key="i">
                    <sup v-if="author.prefix">{{ author.prefix }}</sup>
                    <a class="author-name" v-if="author.homepage" :href="author.homepage" target="_blank">{{ author.name
                        }}</a>
                    <span class="author-name" v-else>{{ author.name }}</span>
                    <sup v-if="author.suffix">{{ author.suffix }}</sup>
                    <span v-if="i < authorsItem.length - 1">&nbsp;&nbsp;&nbsp;&nbsp;</span>
                </span>
            </div>
        </div>

        <div class="res_link">
            <a v-if="resources.pdf" class="button" :href="resources.pdf" target="_blank">
                <i class="iconfont icon-lm-pdf"></i>
                <span>Paper</span>
            </a>
            <a v-if="resources.arxiv" class="button" :href="resources.arxiv" target="_blank">
                <i class="iconfont icon-lm-Arxiv"></i>
                <span>arXiv</span>
            </a>

            <a v-if="resources.github" class="button" :href="resources.github" target="_blank">
                <i class="iconfont icon-lm-github"></i>
                <span>Code</span>
            </a>

            <a v-if="resources.huggingface" class="button" :href="resources.huggingface" target="_blank">
                <i class="iconfont icon-lm-huggingface"></i>
                <span>HuggingFace</span>
            </a>
        </div>

        <video v-if="mainVideo" v-lazy :src="mainVideo" muted loop controls></video>
    </section>
</template>

<script lang="ts" setup>
import { onMounted } from 'vue';

interface Props {
    title?: string,
    subtitle?: string,
    authors?: any[],
    resources?: any,
    mainVideo?: string,
}
const { props } = defineProps<{ props: Props }>()

const title = props.title
const subtitle = props.subtitle
const authors = props.authors
const resources = props.resources
const mainVideo = (props.mainVideo || "").startsWith("assets") ? new URL(`../${props.mainVideo}`, import.meta.url).href : props.mainVideo

onMounted(() => {
    if (title) {
        document.title = title
    }
    if (subtitle) {
        document.title += `: ${subtitle}`
    }
})
</script>

<style lang="scss" scoped>
.title {

    .authors {
        @apply text-center text-lg;

        .thin {
            .author-name {
                @apply font-light
            }
        }
    }

    .res_link {
        @apply text-center mt-1;
    }

    video {
        max-width: 960px;
        @apply mt-4 block w-full;
    }
}

.button {
    @apply mr-3 mt-2;

    i {
        @apply mr-1;
    }
}
</style>
