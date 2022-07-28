# Highlighting utilities

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory collects syntax highlighting utilities for use with Carbon code.

## [highlight.js](https://highlightjs.org/)

This JavaScript syntax highlighting system is used on webpages or in
presentation software like [reveal.js](https://revealjs.com/).

The code is in [highlightjs_carbon_lang.js](highlightjs_carbon_lang.js), and you
can use it by registering it after you load highlight.js:

```html
<script type="module">
    import Carbon from './highlightjs_carbon_lang.js';
    hljs.registerLanguage('Carbon', Carbon);
    hljs.highlightAll();
</script>
```

See [highlightjs_example.html](highlightjs_example.html) for a more complete
example.

It includes many Carbon-specific markup indicators that may be useful to
customize display or highlighting of Carbon code.
