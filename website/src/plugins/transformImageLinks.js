const visit = require('unist-util-visit');

// Adjust HTML image links referring inside the repo to work on the website.
const plugin = (options) => {
  const transformer = (ast) => {
    visit(ast, 'jsx', (node) => {
      node.value = node.value
        .replaceAll(/\bhref="(docs\/[^"]+)"/g, 'href="https://github.com/carbon-language/carbon-lang/blob/trunk/$1"')
        .replaceAll(/\bsrc="(docs\/[^"]+)"/g, 'src="https://raw.githubusercontent.com/carbon-language/carbon-lang/trunk/$1"');
    });
  };
  return transformer;
};

module.exports = plugin;
