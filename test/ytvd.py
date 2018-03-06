import sys
from pytube import YouTube


def download_main(link):
    video_format = 'mp4'
    yt = YouTube(link)
    obj = yt.filter(video_format)[-1]  # select the highest resolution
    resolution = obj.resolution    # obtain the video resolution
    video = yt.get(video_format, resolution)
    video.download('F:\\YouTube_download')


if __name__ == '__main__':
    download_main(sys.argv[1])
    # for link in links_lizhi[11:]:
    #     download_main(link)

    # download_main('https://www.youtube.com/watch?v=s6_sMaCq7Mo')
