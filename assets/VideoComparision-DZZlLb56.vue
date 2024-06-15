<template>
    <section>
        <h3 v-if="title">{{ title }}</h3>
        <h4 v-if="subtitle">{{ subtitle }}</h4>
        <div class="panel">
            <div ref="carouselElement" :id="id" class="relative" data-twe-carousel-init data-twe-carousel-slide
                data-twe-ride="carousel" data-twe-interval="9999999">

                <!--Carousel items-->
                <div class="relative w-full overflow-hidden after:clear-both after:block after:content-['']">
                    <div v-for="(item, i) in items" :key="i" :ref="(el: any) => carouselItems[i] = el"
                        :class="{ hidden: i > 0 }"
                        class="video-group relative float-left -mr-[100%] w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
                        data-twe-carousel-item style="backface-visibility: hidden">
                        <video :ref="(el: any) => videos[0 + i * 2] = el" v-lazy controls :src="item[0]"></video>
                        <div></div>
                        <video :ref="(el: any) => videos[1 + i * 2] = el" v-lazy controls :src="item[1]"></video>
                    </div>
                </div>

                <!--Carousel indicators-->
                <div v-if="items.length > 1"
                    class="absolute bottom-0 left-0 right-0 z-[2] mx-[15%] -mb-8 flex list-none justify-center p-0"
                    data-twe-carousel-indicators>
                    <button v-for="(_item, i) in items" :key="i" :ref="(el: any) => carouselIndicators[i] = el"
                        type="button" :data-twe-target="`#${id}`" :data-twe-slide-to="i" class="indicator"
                        aria-current="true" :aria-label="`Slide ${i + 1}`"></button>
                </div>

                <!--Carousel controls - prev item-->
                <button v-if="items.length > 1" class="indicator-btn indicator-left-btn" type="button"
                    :data-twe-target="`#${id}`" data-twe-slide="prev">
                    <span class="inline-block h-8 w-8">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                            stroke="currentColor" class="h-6 w-6">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
                        </svg>
                    </span>
                    <span
                        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Previous</span>
                </button>
                <!--Carousel controls - next item-->
                <button v-if="items.length > 1" class="indicator-btn indicator-right-btn" type="button"
                    :data-twe-target="`#${id}`" data-twe-slide="next">
                    <span class="inline-block h-8 w-8">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                            stroke="currentColor" class="h-6 w-6">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                        </svg>
                    </span>
                    <span
                        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Next</span>
                </button>
            </div>
        </div>
    </section>

</template>

<script setup lang="ts">
interface Props {
    id?: string
    title?: string
    subtitle?: string
    items?: string[][]
}
const { props } = defineProps<{ props: Props }>()
const title = props.title || ''
const subtitle = props.subtitle || ''
const items = (props.items || []).map(v => v.map(sub => sub.startsWith("assets") ? new URL(`../${sub}`, import.meta.url).href : sub))
const id = props.id || title.replaceAll(" ", "")

import { ref, onMounted } from 'vue';
import { initTWE, Carousel } from 'tw-elements';
import { inVisible } from '@/utils/video';
import { store } from '@/store'

const carouselElement = ref<HTMLElement>();
const videos = ref<HTMLVideoElement[]>([]);
const carouselItems = ref<HTMLElement[]>([]);
const carouselIndicators = ref<HTMLElement[]>([]);

onMounted(async () => {
    carouselItems.value[0]?.setAttribute("data-twe-carousel-active", "")
    carouselIndicators.value[0]?.setAttribute("data-twe-carousel-active", "")
    do {
        await new Promise(resolve => setTimeout(resolve, 100))
    } while (store.tweInitializing["Carousel"])
    store.setInitializing("Carousel", true)
    console.log("initializing..", store.tweInitializing["Carousel"])
    initTWE({ Carousel }, { allowReinits: true, checkOtherImports: true });
    store.setInitializing("Carousel", false)
    console.log("initialized", store.tweInitializing["Carousel"])

    carouselElement.value?.addEventListener('slide.twe.carousel', (v: any) => {
        const from = v.from;
        const to = v.to;
        videos.value[2 * from]?.pause();
        videos.value[2 * from + 1]?.pause();
        // if (inVisible(videos.value[2 * from])) {
        //     videos.value[2 * to].play();
        //     videos.value[2 * to + 1].play();
        // }
    })
});

</script>

<style scoped lang="scss">
section {
    @apply w-full py-10 md:px-16 px-6;
    @apply flex flex-col justify-center items-center;
}

.panel {
    max-width: 960px;
    @apply w-full mt-2;

    &>* {
        @apply w-full mb-8;
    }

    :last-child {
        @apply mb-0;
    }
}

h4 {
    @apply text-lg mb-3
}

.video-group {
    video {
        width: 49%;
        @apply rounded-lg;
    }

    @media (max-width: 768px) {
        video {
            width: 100% !important;
        }

        div {
            width: 0;
        }
    }

    div {
        width: 1%;
    }

    * {
        @apply inline-block;
    }
}
</style>
