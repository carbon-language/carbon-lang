// Get all community links elements
const communityLinks = document.querySelectorAll('ul');

// Add event listeners to each community link
communityLinks.forEach(links => {
    links.addEventListener('click', () => {
        // Get the link id
        const linkId = links.id;

        // Get the corresponding link element
        const link = document.querySelector(`#link-${linkId}`);

        // Show the link
        link.style.display = 'block';
    });
});