<template>
    <section>
        <h3>{{ title }}</h3>
        <div class="panel">
            <div ref="carouselElement" :id="id" class="relative" data-twe-carousel-init data-twe-carousel-slide
                data-twe-ride="carousel" data-twe-interval="9999999">

                <!--Carousel items-->
                <div class="relative w-full overflow-hidden after:clear-both after:block after:content-['']">
                    <div v-for="(videos, i) in videoLists" :key="i" :ref="(el: any) => carouselItems[i] = el"
                        :class="{ hidden: i > 0 }"
                        class="video-group relative float-left -mr-[100%] w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
                        data-twe-carousel-item style="backface-visibility: hidden">
                        <video v-for="(video, vi) in videos" :key="vi" :ref="(el: any) => videos[i + vi] = el" v-lazy
                            controls :src="video"
                            :style="{ width: `${100 / videos.length - 1}%`, 'margin-right': `1%` }"></video>
                    </div>
                </div>

                <!--Carousel indicators-->
                <div v-if="videoLists.length > 1"
                    class="absolute bottom-0 left-0 right-0 z-[2] mx-[15%] -mb-8 flex list-none justify-center p-0"
                    data-twe-carousel-indicators>
                    <button v-for="(_item, i) in videoLists" :key="i" :ref="(el: any) => carouselIndicators[i] = el"
                        type="button" :data-twe-target="`#${id}`" :data-twe-slide-to="i" class="indicator"
                        aria-current="true" :aria-label="`Slide ${i + 1}`"></button>
                </div>

                <!--Carousel controls - prev item-->
                <button v-if="videoLists.length > 1" class="indicator-btn indicator-left-btn" type="button"
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
                <button v-if="videoLists.length > 1" class="indicator-btn indicator-right-btn" type="button"
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
    items?: string[]
    count?: number
}
const { props } = defineProps<{ props: Props }>()
const title = props.title || ''
const items = (props.items || []).map(sub => sub.startsWith("assets") ? new URL(`../${sub}`, import.meta.url).href : sub)
const count = (props.count || 1)
const id = props.id || title.replaceAll(" ", "")

const videoLists: string[][] = []
for (let i = 0; i < items.length; i++) {
    if (i % count === 0) {
        videoLists.push(items.slice(i, count + i))
    }
}

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
