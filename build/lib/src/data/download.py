import requests
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-il', '--image_link',
                    default='https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg',
                    type=str, required=False)
parser.add_argument('-od', '--output_dir',
                    type=str, required=False)

args = parser.parse_args()

url = args.image_link
out_path = os.path.join(args.output_dir, os.path.basename(url))

r = requests.get(url, allow_redirects=True)

open(out_path, 'wb').write(r.content)

