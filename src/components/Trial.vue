<template>
<div>
  <h3 is="sui-header" block>
    A very dull header
  </h3>
    <sui-grid
    :columns="2"
    relaxed="very"
    stackable
    vertical-align="middle"
  >
    <sui-grid-column>
      <div class="ui form">
        <div class="field">
          <label>Username</label>
          <div class="ui left icon input">
            <input
              type="text"
              placeholder="Username"
              autocomplete="off"
              :style="`background-image: url(${image});`"
            >
            <i class="user icon"></i>
          </div>
        </div>
        <div class="field">
          <label>Password</label>
          <div class="ui left icon input">
            <input
              type="password"
              autocomplete="off"
              :style="`background-image: url(${image});`"
            >
            <i class="lock icon"></i>
          </div>
        </div>
        <div class="ui blue submit button">Login</div>
      </div>
    </sui-grid-column>

    <sui-divider vertical>Or</sui-divider>

    <sui-grid-column>

      <div class="center aligned column">
        <div class="ui big green labeled icon button">
          <i class="signup icon"></i>
          Sign Up
        </div>
      </div>
    </sui-grid-column>
  </sui-grid>
</div>
</template>
<script>
export default {
  name: 'Trial',
  data() {
    return {
      msg: 'Welcome MY NAME IS YOMNAAA'
    }
  }
}
</script>
<!-- Add "scoped" attribute to limit CSS to this component only -->
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
</style>