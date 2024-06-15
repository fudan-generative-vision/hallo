import { reactive } from 'vue'
export const store = reactive<{
  tweInitializing: { [key: string]: any }
  setInitializing: (componentName: string, initializing: boolean) => void
}>({
  tweInitializing: {},
  setInitializing(componentName: string, initializing: boolean) {
    this.tweInitializing[componentName] = initializing
    this.tweInitializing = { ...this.tweInitializing }
  }
})
