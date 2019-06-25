<template>
<div>
  <h1 is="sui-header" block>
    I AM READING YOUR MIND
  </h1>
    <sui-grid
    :columns="2"
    relaxed="very"
    stackable
    vertical-align="middle"
  >
    <sui-grid-column>
      <div class="ui card">
          <ImageCard v-bind:prediction="prediction"></ImageCard>
            <sui-card-content>
              <sui-card-description>
               The user is moving his {{prediction}}
              </sui-card-description>
            </sui-card-content>
      </div>
    </sui-grid-column>

    <sui-grid-column>
      <SettingsForm v-on:sendPrediction="getPrediction"></SettingsForm>
      <TimerApp>
      </TimerApp>
    </sui-grid-column>

  </sui-grid>
</div>
</template>

<script>
import ImageCard from './ImageCard'
import SettingsForm from './SettingsForm'
import TimerApp from './TimerApp'
export default {
  name: 'interface_app',
  components: {
      ImageCard,
      SettingsForm,
      TimerApp,
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
h1{
  font-weight: normal;
  color: #596B80;
  letter-spacing: 3px;
}
h2 {
  font-weight: normal;
  color: #596B80;
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
  margin-left : 25%;
  width: 60%;
}
.block{
  margin-top:0%;
  padding-top: 0px;
  margin-bottom:20px;
  position: relative;
  top:-50px;
  color: #596B80;
}
.button{
  background-color:#596B80;
  color: white;
}
.ui.card>.content{
  background-color:#596B80;
}
.ui.card>.content>.description {
  clear: both;
  color: white;
  font-weight: bold;
  font-size: larger
}
</style>
