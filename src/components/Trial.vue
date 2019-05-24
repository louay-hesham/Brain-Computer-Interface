<template>
<div>
  <h3 is="sui-header" block>
    I AM READING YOUR MIND
  </h3>
    <sui-grid
    :columns="2"
    relaxed="very"
    stackable
    vertical-align="middle"
  >

    <sui-grid-column>
      <div class="ui card">
          <ImageCard v-bind:prediction="prediction"></ImageCard>
          <div class="ui two buttons">
          <button class="ui button" type="submit">Start</button>
          </div>
      </div>
    </sui-grid-column>

    <sui-grid-column>
      <SettingsForm v-on:sendPrediction="getPrediction"></SettingsForm>
    </sui-grid-column>
    <!-- <div class="clock">
    </div> -->
  </sui-grid>
</div>
</template>

<script>
import ImageCard from './ImageCard'
import SettingsForm from './SettingsForm'
export default {
  name: 'Trial',
  components: {
      ImageCard,
      SettingsForm
      },
  data() {
    return {
      // start at the very begining with the man standing
      prediction: -1
    }
  },
  methods: {
    getPrediction (prediction){
      this.prediction = prediction
    }
  }
}
</script>

<style scoped>
h1,
h2 {
  font-weight: normal;
  color: #596B80;
}
h3{
  color: #F27584;
  letter-spacing: 3px;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
.card{
  margin-left : 20%;
  width: 60%;

}
.block{
  margin-top:0%;
  padding-top: 0px;
  margin-bottom:20 px;
  position: relative;
  top:-50px;
  color: #596B80;
}
.form{
  margin-right : 10%;
  position: relative;
  top:-100px;
}
.button{
  background-color:#596B80;
  color: white;
}
.clock{
  position: absolute;
  top: 30%;
  left:45%;
  transform: translate(-50%,-50%);
  width: 150px;
  height:150px;
  border: 10px solid #F27584;
  border-radius: 50%;
  background: #fff;
  box-shadow: -2px 2px 0 #e23232, inset 0 0 20px (0,0,0,0,0.5);
}
.clock:before{
  content:'';
  position: absolute;
  top: 48% ;
  left:50%;
  width: 40%;
  height:6px;
  background: #262626;
  border-radius: 3px;
  animation: animate 10s linear infinite;
  transform-origin: left;
}
@keyframes animate{
  0%{
    transform: rotate(0deg);
  }
  100%
  {
    transform: rotate(360deg)
  }
}
</style>
