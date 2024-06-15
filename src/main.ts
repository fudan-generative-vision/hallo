import '@/assets/main.css'
import '@/assets/carousel.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import videoLazy from './directives/video-lazy'

const app = createApp(App)
app.use(router)

app.directive(videoLazy.name, videoLazy.option)

app.mount('#app')
