# Full list of configuration options available here:
# https://github.com/hakimel/reveal.js#configuration
do ->


  Reveal.initialize
    controls: false
    progress: false
    slideNumber: false
    history: true
    center: false
    maxScale: 1.5
    transition: 'linear'
    transitionSpeed: 'default'
    fragments: true
    theme: Reveal.getQueryHash().theme
    math: mathjax: 'https://cdn.mathjax.org/mathjax/latest/MathJax.js'
    multiplex:
      secret: SECRET
      id: '80519760bbc84258'
      url: 'https://reveal-js-multiplex-ccjbegmaii.now.sh'

    dependencies: [
      {
        src: 'lib/js/classList.js'
        condition: ->
          !document.body.classList
      }
      {
        src: '//cdn.socket.io/socket.io-1.3.5.js'
        async: true
      }
      {
        src: 'plugin/multiplex/master.js'
        async: true
      }
      {
        src: "plugin/multiplex/#{if SECRET? then 'master' else 'client'}.js"
        async: true
      }
      {
        src: 'plugin/highlight/highlight.js'
        async: true
        callback: ->
          hljs.initHighlightingOnLoad()
          return
      }
      {
        src: 'plugin/zoom-js/zoom.js'
        async: true
      }
      {
        src: 'plugin/notes/notes.js'
        async: true
      }
      {
        src: 'plugin/math/math.js'
        async: true
      }
    ]
  # Set backgorund opacity
  sections = document.getElementsByTagName('section')
  backgrounds = document.getElementsByClassName('slide-background')
  i = 0
  while i < sections.length
    section = sections[i]
    opacity = section.getAttribute('data-background-opacity')
    if !opacity
      i++
      continue
    background = backgrounds[i]
    backgroundStyle = background.getAttribute('style')
    if !backgroundStyle
      backgroundStyle = ''
    backgroundStyle = backgroundStyle + ';opacity: ' + opacity + ';'
    background.setAttribute 'style', backgroundStyle
    i++

  Reveal.configure {keyboard: {13: 'next'}}


