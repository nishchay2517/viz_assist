import json
import csv

def json_to_txt(input_file, output_file):
    with open(input_file, 'r',encoding='utf-8') as json_file:
        data = json.load(json_file)

        images = data['images']
        annotations = data['annotations']
        
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            txt_file.write("image,caption\n")
            for image, annotation in zip(images, annotations):
                txt_file.write(f"{image['file_name']},\"{annotation['caption']}\"\n")

def json_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    images = data['images']
    annotations = data['annotations']

    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'caption'])  # Write the header

        for image, annotation in zip(images, annotations):
            writer.writerow([image['file_name'], '"{}"'.format(annotation['caption'])])


# Example usage
input_file = 'annotations.json'
output_file = 'data.txt'
json_to_txt(input_file, output_file)

output_file1 = 'data.csv'
json_to_csv(input_file, output_file1)


