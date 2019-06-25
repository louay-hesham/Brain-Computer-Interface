<template>
  <div>
    <!-- our template -->
    <section id="app" >
    <div>
      <div id="timer">
        <span id="minutes">{{ minutes }}</span>
        <span id="middle">:</span>
        <span id="seconds">{{ seconds }}</span>
      </div>

      <div id="buttons">
    <!--     Start TImer -->
        <!-- <button
          id="start"
          class="ui button"
          v-if="!timer"
          @click="startTimer">
            <i>Start</i>
        </button> -->
    <!--     Pause Timer -->
        <!-- <button
          id="stop"
          class="ui button"
          v-if="timer"
          @click="stopTimer">
            <i>Stop</i>
        </button> -->
    <!--     Restart Timer -->
        <!-- <button
          id="reset"
          class="ui button"
          v-if="resetButton"
          @click="resetTimer">
            <i>Reset</i>
        </button> -->
      </div>
    </div>
    </section>
  </div>
</template>

<script>
export default {
  name: 'timerApp',
  data( ) {
    return{
      timer: null,
      totalTime: 0,
      resetButton: false,
    }
  },
   props: ['delay'],
  methods: {
    startTimer: function() {
      this.totalTime = this.delay
      this.timer = setInterval(() => this.countdown(), 1000);
      this.resetButton = true;
    },
    stopTimer: function() {
      clearInterval(this.timer);
      this.timer = null;
      this.resetButton = true;
    },
    resetTimer: function() {
      this.totalTime = this.delay;
      clearInterval(this.timer);
      this.timer = null;
      this.resetButton = false;
    },
    padTime: function(time) {
      if (time < 10){
        return (time < this.delay ? '0' : '') + time;
      }
      else return time;
    },
    countdown: function() {
      if (this.totalTime ==0){
        this.resetTimer();
      }
      else {
        this.totalTime--;
      }
    }
  },
  computed: {
    minutes: function() {
      const minutes = Math.floor(this.totalTime / 60);
      return this.padTime(minutes);
    },
    seconds: function() {
      const seconds = this.totalTime - (this.minutes * 60);
      return this.padTime(seconds);
    }
  }
}
</script>

<style scoped>
#message {
  color: #DDD;
  font-size: 10px;
  margin-bottom: 10px;
  margin-right:25%;
}

#timer {
  font-size: 50px;
  line-height: 1;
  margin-right:20%;
  margin-top:-40px;
}
.timer-title{
  margin-right:25%;
}
.button{
  background-color:#596B80;
  color: white;
  margin-top:10px;
  margin-right:20%;
}
</style>
