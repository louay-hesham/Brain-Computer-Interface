<template>
  <div class="image">
    <img src="../assets/standing.jpg" v-if="prediction == '...'">
    <img src="../assets/RightHand.jpg" v-if="prediction == 'right_hand'">
    <img src="../assets/LeftHand.jpg" v-if="prediction == 'left_hand'">
    <img src="../assets/EyesClosed2.jpg" v-if="prediction == 'eye_closed'">
    <img src="../assets/BothFeets2.jpg" v-if="prediction== 'both_feet'">
    <img src="../assets/BothHands.jpg" v-if="prediction== 'both_hands'">
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
