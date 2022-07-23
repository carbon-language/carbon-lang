// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Carbon Language',
  tagline: 'An experimental successor to C++',
  url: 'https://carbon-language.github.io',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/carbon-logo.png',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'carbon-language',
  projectName: 'carbon-lang',

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
          sidebarPath: undefined,
          editUrl: 'https://github.com/carbon-language/carbon-lang/blob/trunk/',
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
      navbar: {
        title: 'Carbon Language',
        logo: {
          alt: 'Carbon Logo',
          src: 'img/carbon-logo.png',
        },
        items: [
          {
            to: '/docs',
            label: 'Docs',
            position: 'left'
          },
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
              }
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
