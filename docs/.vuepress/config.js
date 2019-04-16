module.exports = {
  base: '/',
  head: [
    ['link', { rel: 'icon', href: '/logo.png' }]
  ],
  locales: {
    '/': {
      lang: 'en-US',
      title: 'HTML5 & Friends',
      description: 'It is a new version of the language HTML, with new elements, attributes, and behaviors, and a larger set of technologies that allows the building of more diverse and powerful Web sites and applications.'
    },
    '/zh/': {
      lang: 'zh-CN',
      title: 'HTML5技术集',
      description: 'HTML5 允许更多样化和强大的网站和应用程序'
    }
  },
  themeConfig: {
    locales: {
      '/': {
        selectText: 'Languages',
        label: 'English',
        nav: [
          { text: 'Guide', link: '/guide/' },
          { text: 'External', link: 'https://vuepress.vuejs.org/' }
        ]
      },
      '/zh/': {
        selectText: '选择语言',
        label: '简体中文',
        nav: [
          { text: '指南', link: '/zh/guide/' },
          { text: 'AI', link: '/zh/tensorflow/' },
          { text: '扩展', link: 'https://vuepress.vuejs.org/zh/' }
        ],
        sidebar: {
          '/zh/guide/': [{
            title: '指南',
            collapsable: false,
            children: ['', 'vue', 'webpack', 'pwa']
          }],
          '/zh/tensorflow/': [{
            title: 'AI',
            collapsable: false,
            children: ['']
          }]
        }
      }
    }
  }
}
