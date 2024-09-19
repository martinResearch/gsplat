import requests
import os
import argparse

# Automatically get the repository name in the format "owner/repo" from the GitHub workflow environment
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")

def list_python_wheels():
    # GitHub API URL for releases
    releases_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases"

    response = requests.get(releases_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch releases: {response.status_code} {response.text}")

    releases = response.json()

    wheel_files = []

    # Iterate through releases and assets
    for release in releases:
        assets = release.get("assets", [])
        for asset in assets:
            if asset["name"].endswith(".whl"):
                wheel_files.append({
                    "release_name": release["name"],
                    "wheel_name": asset["name"],
                    "download_url": asset["browser_download_url"]
                })

    return wheel_files

def generate_simple_index(wheels):
    html_content = """
    <!DOCTYPE html>
    <html>
      <head><title>Links for {repo}</title></head>
      <body>
        <h1>Links for {repo}</h1>
    """.format(repo=GITHUB_REPO)

    # Add links for each wheel
    for wheel in wheels:
        html_content += f'<a href="{wheel["download_url"]}">{wheel["wheel_name"]}</a><br/>\n'

    html_content += """
      </body>
    </html>
    """

    return html_content


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate Python Wheels Index Pages")
    argparser.add_argument("--outdir", help="Output directory for the index pages", default=".")
    args = argparser.parse_args()
    wheels = list_python_wheels()
    if wheels:
        print("Python Wheels found in releases:")
        for wheel in wheels:
            print(f"Release: {wheel['release_name']}, Wheel: {wheel['wheel_name']}, URL: {wheel['download_url']}")
    else:
        print("No Python wheels found in the releases.")

    # Generate Simple Index HTML
    html_page = generate_simple_index(wheels)

    # Save the HTML content to a file in the output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "index.html"), "w") as file:
        file.write(html_page)

