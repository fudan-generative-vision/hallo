export function inVisible(video: HTMLElement) {
  const { left, right, top, bottom, width, height } = video.getBoundingClientRect()
  if (bottom < 0 || top > window.innerHeight) {
    return false
  } else if (left != 0 && right != 0) {
    return true
  }
  return false
}
