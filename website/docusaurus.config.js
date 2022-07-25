// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const transformLinks = require('./src/plugins/transformLinks');
const transformImageLinks = require('./src/plugins/transformImageLinks');
const removeTableOfContents = require('./src/plugins/removeTableOfContents');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Carbon Language',
  tagline: 'An experimental successor to C++',
  url: 'https://carbon-language.github.io',
  baseUrl: '/',
  onBrokenLinks: 'ignore', // TODO: Fix broken links, and change this back to 'throw'
  onBrokenMarkdownLinks: 'ignore', // These are fixed in `transformLinks`.
  favicon: 'img/carbon-logo.png',

  // GitHub pages deployment config.
  organizationName: 'carbon-language',
  projectName: 'carbon-language.github.io',
  trailingSlash: true, // Fixes relative directory links on e.g. Spec page.

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '..',
          routeBasePath: '/',
          include: ['docs/**/*.{md,mdx}', 'README.md'],
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/carbon-language/carbon-lang/blob/trunk/docs',
          beforeDefaultRemarkPlugins: [removeTableOfContents, transformLinks],
          rehypePlugins: [transformImageLinks],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
  /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      announcementBar: {
        content: '⚠️ Carbon is experimental and not production-ready, see ' +
          '<a href="/docs/project/faq#how-soon-can-we-use-carbon">How soon can we use Carbon?</a>',
        backgroundColor: 'var(--ifm-navbar-background-color)',
        textColor: 'var(--ifm-font-color-base)',
        isCloseable: true,
      },
      navbar: {
        title: 'Carbon Language',
        logo: {
          alt: 'Carbon Logo',
          src: 'img/carbon-logo.png',
        },
        items: [
          { to: '/docs/design', label: 'Design', position: 'left' },
          { to: '/docs/project', label: 'Project', position: 'left' },
          { to: '/docs/guides', label: 'Guides', position: 'left' },
          { to: '/docs/spec', label: 'Spec', position: 'left' },
          {
            href: 'https://github.com/carbon-language/carbon-lang',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/carbon-language/carbon-lang',
              },
              {
                label: 'Discord',
                href: 'https://discord.gg/ZjVdShJDAs',
              },
              {
                label: 'Code of Conduct',
                href: 'https://github.com/carbon-language/carbon-lang/blob/trunk/CODE_OF_CONDUCT.md',
              },
            ],
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
