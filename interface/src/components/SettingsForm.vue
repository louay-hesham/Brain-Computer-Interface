<template>
    <div>
        <form class="ui form" @submit="settingsSubmit">
        <h1>Settings</h1>
        <div class="field">
        <label>Number of samples</label>
        <input type="text" name="number-of-samples" placeholder="Number of samples" v-model="numberOfSamples">
      </div>
      <div class="field">
        <label>Frequency</label>
        <input type="text" name="frequency" placeholder="Frequency in HZ" v-model="frequency">
      </div>
      <div class="field">
        <label>Delay</label>
        <input type="text" name="delay" placeholder="Delay in seconds" v-model="delay">
      </div>
      <div>{{prediction}}</div>
      <button class="ui button" type="submit">Submit</button>
    </form>
    </div>
</template>
<script>
export default {
    name: 'settingsForm',
    data() {
       return {
      frequency: '',
      delay: '',
      numberOfSamples: '',
      prediction: -1
    }
  },
  methods: {
      settingsSubmit(e){
        e.preventDefault();
        this.axios.post('http://127.0.0.1:8000/predict/',{
          samples_count: this.numberOfSamples,
          freq: this.frequency,
          delay: this.delay
        })
        .then((response)=> {
          this.prediction = response.data.prediction
          this.$emit('sendPrediction', this.prediction)
        })
        .catch((error)=> {
          console.log(error)
        })
      }
    }
  }
</script>

<style scoped>
h1,
h2 {
  font-weight: bold;
  color: #596B80;
}
h3{
  color: #F27584;
  letter-spacing: 3px;
  font-size: 50px;
}
ul {
  list-style-type: none;
  padding: 0;
}
a {
  color: #42b983;
}
.form{
  margin-right : 20%;
  position: relative;
  top:-30px;
}
.button{
  background-color:#596B80;
  color: white;
}
.field>label{
  color: white;
  font-size:30px;
  background-color: #596B80;
  text-align: left;
  padding-left:5px;
  padding-top:2px;
  padding-bottom:2px;
}
.ui.form .field>label {
    color: white;
    font-size: 14px;
}
</style>
