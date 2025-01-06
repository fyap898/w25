# Rewrite the README out of laziness
echo "# COMP 4630: Machine Learning" > README.md
echo "## Lecture Quick Links" >> README.md
echo "" >> README.md
echo "| Lecture Number | Title | Source | PDF |" >> README.md
echo "| --- | --- | --- | --- |" >> README.md

# loop through and only process files with the "marp" attribute
for md in lectures/**/*.md; do
    if grep -q "marp:\s*true" $md; then
        doc=$(basename "$md" .md)
        npx @marp-team/marp-cli@latest --theme lectures/marp-theme.css --allow-local-files --pdf --html $md -o lectures/pdfs/$doc.pdf
        npx @marp-team/marp-cli@latest --theme lectures/marp-theme.css --allow-local-files --bespoke.progress --html $md -o lectures/html/$doc.html

        # Add the relevant links to the main README page
        num=${doc:0:2}
        title=$(grep -oP "(?<=^title:\s).+$" $md | xargs)

        echo "| $num | $title | [md]($md) | <a href=\"lectures/pdfs/$doc.pdf\"><img src=\"lectures/figures/file-type-pdf.svg\" alt=\"PDF\" style=\"height: 20px; margin: 0;\"/></a> |" >> README.md
    fi
done
