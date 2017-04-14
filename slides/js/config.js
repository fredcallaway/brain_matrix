// Full list of configuration options available here:
// https://github.com/hakimel/reveal.js#configuration


Reveal.initialize({
    controls: false,         // Display controls in the bottom right corner
    progress: false,         // Display a presentation progress bar
    slideNumber: false,                // Display the page number of the current slide
    history: true,          // Push each slide change to the browser history
    center: false,                       // Vertical centering of slides
    maxScale: 1.5,                  // Bounds for smallest/largest possible content scale
    transition: 'linear' , // default/cube/page/concave/zoom/linear/fade/none
    transitionSpeed: 'default',
    fragments: true,
    theme: Reveal.getQueryHash().theme, // available themes are in /css/theme

    // Optional libraries used to extend on reveal.js
    math: {
        mathjax: 'https://cdn.mathjax.org/mathjax/latest/MathJax.js',
        // config: 'TeX-AMS_HTML-full'  // See http://docs.mathjax.org/en/latest/config-files.html
    },

    dependencies: [
            // Cross-browser shim that fully implements classList - https://github.com/eligrey/classList.js/
            { src: 'lib/js/classList.js', condition: function() { return !document.body.classList; } },

            // Interpret Markdown in <section> elements
            // { src: 'plugin/markdown/marked.js', condition: function() { return !!document.querySelector( '[data-markdown]' ); } },
            // { src: 'plugin/markdown/markdown.js', condition: function() { return !!document.querySelector( '[data-markdown]' ); } },

            // Syntax highlight for <code> elements
            { src: 'plugin/highlight/highlight.js', async: true, callback: function() { hljs.initHighlightingOnLoad(); } },

            // Zoom in and out with Alt+click
            { src: 'plugin/zoom-js/zoom.js', async: true },

            // Speaker notes
            { src: 'plugin/notes/notes.js', async: true },

            // MathJax
            { src: 'plugin/math/math.js', async: true }
        ]
});

    // Set backgorund opacity
      var sections = document.getElementsByTagName("section")
      var backgrounds = document.getElementsByClassName("slide-background")

      for (i = 0; i < sections.length; i++) {
        var section = sections[i]
        var opacity = section.getAttribute("data-background-opacity")
        if (!opacity) { continue; }

        var background = backgrounds[i]
        var backgroundStyle = background.getAttribute("style")

        if (!backgroundStyle) { backgroundStyle = "" }

        backgroundStyle = backgroundStyle + ";opacity: " + opacity + ";"
        background.setAttribute("style", backgroundStyle)
      }

