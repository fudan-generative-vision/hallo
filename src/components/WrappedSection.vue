<template>
    <TitleSection v-if="template === 'title'" :props="props"></TitleSection>
    <AbstractSection v-if="template === 'abstract'" :props="props"></AbstractSection>
    <FrameworkSection v-if="template === 'framework'" :props="props" />
    <VideoComparision v-if="template === 'video-comparision'" :props="props" />
    <TextVideo v-if="template === 'text-video'" :props="props" />
    <SingleVideo v-if="template === 'single-video'" :props="props" />
    <VideoCarousel v-if="template === 'video-carousel'" :props="props" />
    <BibTeX v-if="template === 'bibtex'" :props="props" />
</template>

<script setup lang="ts">
import TitleSection from "@/components/TitleSection.vue"
import AbstractSection from "@/components/AbstractSection.vue"
import FrameworkSection from "@/components/FrameworkSection.vue"
import VideoComparision from "@/components/VideoComparision.vue"
import TextVideo from "@/components/TextVideo.vue"
import SingleVideo from "@/components/SingleVideo.vue"
import VideoCarousel from "@/components/VideoCarousel.vue"
import BibTeX from "@/components/BibTeX.vue"
interface Props {
    template?: string,
    props?: any,
}
const { template, props } = defineProps<Props>();
</script>
