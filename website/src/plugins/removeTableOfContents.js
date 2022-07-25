const visit = require('unist-util-visit');

// Remove table-of-contents enclosed in `<!-- toc -->` and `<!-- tocstop -->` comments.
const plugin = (options) => {
  const transformer = (ast) => {
    visit(ast, 'root', (node) => {
      const tocStart = node.children.findIndex(child => child.type === 'comment' && child.value.match(/^\s*\btoc\b\s*$/));
      const tocEnd = node.children.findIndex(child => child.type === 'comment' && child.value.match(/^\s*\btocstop\b\s*$/));
      if (tocStart !== -1 && tocEnd !== -1) {
        node.children.splice(tocStart, tocEnd - tocStart + 1);
      }
    });
  };
  return transformer;
};

module.exports = plugin;
