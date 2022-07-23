const visit = require('unist-util-visit');

// Adjust Markdown links referring inside the repo to work on the website.
const plugin = (options) => {
  const transformer = (ast) => {
    visit(ast, 'link', (node) => {
      if (node.url.startsWith('/')) {
        if (node.url.match(/^\/docs\/(design|spec|project|guides)/)) {
          // It refers to a doc that exists on the website, change it to the correct path on the website.
          node.url = node.url.replace(/^\/docs/, '').replace(/(\/README)?\.md/, '')
        } else if (!node.url.match(/^\/(design|spec|project|guides)/)) {
          // It refers to a doc that doesn't exist on the website, change it to an external URL referring to the repo.
          node.url = 'https://github.com/carbon-language/carbon-lang/blob/trunk' + node.url;
        }
      }
    });
  };
  return transformer;
};

module.exports = plugin;
