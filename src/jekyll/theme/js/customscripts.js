$('#mysidebar').height($('.nav').height());

$(document).ready(function () {
  // This handles the automatic toc. Use ## for subheads to auto-generate the
  // on-page minitoc. If you use html tags, you must supply an ID for the
  // heading element in order for it to appear in the minitoc.
  $('#toc').toc({
    minimumHeaders: 0,
    listType: 'ul',
    showSpeed: 0,
    headers: 'h2,h3,h4'
  });

  // This script says, if the height of the viewport is greater than 800px,
  // then insert affix class, which makes the nav bar float in a fixed
  // position as your scroll. If you have a lot of nav items, this height may
  // not work for you.
  var h = $(window).height();
  if (h > 800) {
    $('#mysidebar').attr('class', 'nav affix');
  }
  // Activate tooltips. Although this is a bootstrap js function, it must be
  // activated this way in your theme.
  $('[data-toggle="tooltip"]').tooltip({
    placement: 'top'
  });
});

document.addEventListener('DOMContentLoaded', function (event) {
  // If the markdown has its own TOC, hide it.
  const content = document.getElementById('post-content');
  if (content) {
    for (var i = 0; i < content.children.length; ++i) {
      // Scan for a TOC.
      var child = content.children[i];
      if (child.outerHTML != '<h2>Table of contents</h2>') continue;

      // Starting with the TOC header, elements until the next header.
      child.style.display = 'none';
      for (++i; i < content.children.length; ++i) {
        child = content.children[i];
        if (child.tagName.match(/H[0-9]/)) break;
        child.style.display = 'none';
      }
      break;
    }
  }

  // AnchorJS
  // Note that this needs to be run early, not on document.ready(), or jumps
  // to the invented IDs will not work on page load.
  anchors.add('h2,h3,h4,h5');
});

// Needed for nav tabs on pages. See Formatting > Nav tabs for more details.
// Script from http://stackoverflow.com/questions/10523433/how-do-i-keep-the-current-tab-active-with-twitter-bootstrap-after-a-page-reload
$(function () {
  var json, tabsState;
  $('a[data-toggle="pill"], a[data-toggle="tab"]').on('shown.bs.tab', function (
    e
  ) {
    var href, json, parentId, tabsState;

    tabsState = localStorage.getItem('tabs-state');
    json = JSON.parse(tabsState || '{}');
    parentId = $(e.target)
      .parents('ul.nav.nav-pills, ul.nav.nav-tabs')
      .attr('id');
    href = $(e.target).attr('href');
    json[parentId] = href;

    return localStorage.setItem('tabs-state', JSON.stringify(json));
  });

  tabsState = localStorage.getItem('tabs-state');
  json = JSON.parse(tabsState || '{}');

  $.each(json, function (containerId, href) {
    return $('#' + containerId + ' a[href=' + href + ']').tab('show');
  });

  $('ul.nav.nav-pills, ul.nav.nav-tabs').each(function () {
    var $this = $(this);
    if (!json[$this.attr('id')]) {
      return $this
        .find('a[data-toggle=tab]:first, a[data-toggle=pill]:first')
        .tab('show');
    }
  });
});
