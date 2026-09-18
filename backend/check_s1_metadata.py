import glob
import xml.etree.ElementTree as ET


# Find Sentinel-1 VV annotation
annotation = glob.glob(
    r"data/sentinel1/**/*.SAFE/annotation/*vv*.xml",
    recursive=True
)[0]

print("Annotation file:")
print(annotation)


# Read XML
root = ET.parse(annotation).getroot()


print("\n========== SENTINEL-1 METADATA ==========")


# Print important elements
wanted = [
    "missionId",
    "productFirstLineUtcTime",
    "productLastLineUtcTime",
    "mode",
    "swath",
    "polarisation",
    "incidenceAngleMidSwath",
    "incidenceAngleMidSwath",
]


for element in root.iter():

    tag = element.tag.split("}")[-1]

    if tag in wanted:

        if element.text:
            print(
                tag,
                ":",
                element.text.strip()
            )


# Search for incidence angle information
print("\n========== INCIDENCE ANGLE VALUES ==========")

count = 0

for element in root.iter():

    tag = element.tag.split("}")[-1]

    if "incidenceAngle" in tag.lower():

        if element.text:

            text = element.text.strip()

            if len(text) < 100:

                print(
                    tag,
                    ":",
                    text
                )

                count += 1

                if count >= 20:
                    break


print("\n============================================")