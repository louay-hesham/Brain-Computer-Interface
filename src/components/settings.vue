<templete>
  <div>
    <sui-divider vertical>Or</sui-divider>
    <sui-grid-column>
      <form class="ui form">
        <h1>Settings</h1>
      <div class="field">
        <label>Number of samples</label>
        <input type="text" name="number-of-samples" placeholder="Number of samples">
      </div>
      <div class="field">
        <label>Frequency</label>
        <input type="text" name="frequency" placeholder="Frequency in HZ">
      </div>
      <div class="field">
        <label>Delay</label>
        <input type="text" name="delay" placeholder="Delay in seconds">
      </div>
      <button class="ui button" type="submit">Submit</button>
    </form>

    </sui-grid-column>
    </sui-grid>
  </div>
</template>

<script>
export default {
  name: 'settings',
  data() {
    return {
      msg: 'Welcome MY NAME IS Toka'
    }
  }
}
</script>

<style scoped>
h1,
h2 {
  font-weight: normal;
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
  margin-left : 15%;
  width: 60%;

}

.form{
  margin-right : 15%;
  position: relative;
  top:-100px;
}

</style>
