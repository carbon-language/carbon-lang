const visit = require('unist-util-visit');

// Adjust Markdown links referring inside the repo to work on the website.
const plugin = (options) => {
  const transformer = (ast) => {
    visit(ast, 'link', (node) => {
      if ((node.url.startsWith('/') && !node.url.match(/^\/docs\/(design|spec|project|guides)/)) || node.url.includes('.md')) {
        // It refers to a doc that doesn't exist on the website, change it to an external URL referring to the repo.
        node.url = 'https://github.com/carbon-language/carbon-lang/blob/trunk/' + node.url.replace(/^\//, '');
      }
    });
  };
  return transformer;
};

module.exports = plugin;
