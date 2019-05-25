<template>
  <div class="image">
    <img src="../assets/standing.jpg" v-if="prediction == -1">
    <img src="../assets/RightHand.jpg" v-if="prediction == 2">
    <img src="../assets/LeftHand.jpg" v-if="prediction == 3">
    <img src="../assets/EyesClosed2.jpg" v-if="prediction == 4">
    <img src="../assets/BothFeets.jpg" v-if="prediction== 5">
  </div>
</template>
<script>
export default {
  name: 'imageCard',
  props: ['prediction']
}
</script>

<style scoped>
.image{
  width:300px;
  height:auto;
  margin-left:10%;
  margin-right:30%;
  margin-top:2%;
  margin-bottom:2%;
}
</style>
