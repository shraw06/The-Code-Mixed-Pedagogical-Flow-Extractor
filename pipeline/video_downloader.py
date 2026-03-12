import yt_dlp as youtube_dl

with open('data/video_links.txt', 'r') as f:
    urls = f.readlines()
    urls = [url.strip() for url in urls]

ydl_opts = {
        'format': 'best',
        'outtmpl': 'data/videos/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
    }

for url in urls:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])