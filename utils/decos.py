from .std import *
import openai

'''
APIConnectionError	Cause: Issue connecting to our services. RETRY
Solution: Check your network settings, proxy configuration, SSL certificates, or firewall rules.

APITimeoutError	Cause: Request timed out. RETRY
Solution: Retry your request after a brief wait and contact us if the issue persists.

AuthenticationError	Cause: Your API key or token was invalid, expired, or revoked. DELETE_AND_TRY_NEXT
Solution: Check your API key or token and make sure it is correct and active. You may need to generate a new one from your account dashboard.

BadRequestError	Cause: Your request was malformed or missing some required parameters, such as a token or an input. ERROR
Solution: The error message should advise you on the specific error made. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. You may also need to check the encoding, format, or size of your request data.

ConflictError	Cause: The resource was updated by another request. ERROR
Solution: Try to update the resource again and ensure no other requests are trying to update it.

InternalServerError	Cause: Issue on our side. RETRY
Solution: Retry your request after a brief wait and contact us if the issue persists.

NotFoundError	Cause: Requested resource does not exist. ERROR
Solution: Ensure you are the correct resource identifier.

PermissionDeniedError	Cause: You don't have access to the requested resource. TRY_NEXT
Solution: Ensure you are using the correct API key, organization ID, and resource ID.

RateLimitError	Cause: You have hit your assigned rate limit. TRY_NEXT
Solution: Pace your requests. Read more in our Rate limit guide.

UnprocessableEntityError	Cause: Unable to process the request despite the format being correct. RETRY
Solution: Please try the request again.

'''

class NoValidAPIKey(Exception):
    pass

class TryAPIKeysUntilSuccess:
    def __init__(self, api_keys=[], remove_bad_api_keys=False) -> None:
        self.api_keys = api_keys
        self.remove_bad_api_keys = remove_bad_api_keys

    def __call__(self, func: Callable):
        def res_func(*args, **kwargs):
            if 'api_key' in kwargs.keys():
                api_keys: List[str] = kwargs['api_key']
            else:
                api_keys = self.api_keys
            items_to_delete = []
            for i, api_key in enumerate(api_keys):
                success = False
                while True:
                    todo = 'SUCCESS'

                    kwargs2 = kwargs.copy()
                    kwargs2['api_key'] = api_key

                    try:
                        res = func(*args, **kwargs2)
                    except openai.APIConnectionError or openai.APITimeoutError or openai.InternalServerError or openai.UnprocessableEntityError as e:
                        todo = 'RETRY'
                    except openai.AuthenticationError:
                        todo = 'DELETE_AND_TRY_NEXT'
                        items_to_delete.append(i)
                    except openai.PermissionDeniedError or openai.RateLimitError:
                        todo = 'TRY_NEXT'
                    
                    if todo == 'SUCCESS':
                        success = True
                        break
                    if todo in ['DELETE_AND_TRY_NEXT', 'TRY_NEXT']:
                        break
                if success:
                    break
            
            if self.remove_bad_api_keys:
                items_to_delete.reverse()
                for i in items_to_delete:
                    api_keys.pop(i)
            
            if success:
                return res
            else:
                raise NoValidAPIKey()
        
        return res_func
                    