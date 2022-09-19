# Programmatic way to generate `topic_detection.css`

css = ".istopic {\n" \
      "color: #6b2bd6;" \
      "\n}" \
      "\n\n"

# Font size of highest level topic
starting_fs = 30
# Font size difference between topic and subtopic
fs_diff = 5
# Minimum font size of text
fs_min = 15
# Number of pixels to indent at each level
ind = 18

for i in range(10):
    css += f".topic-L{i} {{\n" \
           f"font-size: {max(starting_fs-i*fs_diff, fs_min)}px;\n" \
           f"text-indent: {ind*i}px;\n" \
           f"}}" \
           f"\n\n"

with open('topic_detection.css', 'w') as f:
    f.write(css)
