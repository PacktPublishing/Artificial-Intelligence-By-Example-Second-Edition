#Google Translate
#Built with Google Translation tools
#Copyright 2019 Denis Rothman MIT License. See LICENSE.

from googleapiclient.discovery import build
import html

def g_translate(source,targetl):
    service = build('translate', 'v2',developerKey='[YOUR API KEY]')
    request = service.translations().list(q=source, target=targetl)
    response = request.execute()
    return response['translations'][0]['translatedText']

source='Google Translate is great!'
targetl="fr"  
result = g_translate(source,targetl)
print("result:", html.unescape(result))


