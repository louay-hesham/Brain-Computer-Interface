import Vue from 'vue'
import App from './App.vue'
import SuiVue from 'semantic-ui-vue'
import axios from 'axios'
import VueAxios from 'vue-axios'
Vue.use(SuiVue);
Vue.use(VueAxios, axios)
Vue.config.productionTip = false
new Vue({
  el: '#app',
  render: h => h(App)
})
