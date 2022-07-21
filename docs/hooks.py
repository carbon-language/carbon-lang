def on_page_content(html, page, config, files):
    return html.replace('href="/', 'href="https://github.com/carbon-language/carbon-lang/tree/trunk/')
