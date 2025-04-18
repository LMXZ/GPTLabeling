from .std import *
import openai
from .shifter import Shifter

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

class TooManyRetries(Exception):
    def __str__(self):
        return "Too many retries!!"

class TryAPIKeysUntilSuccess:
    def __init__(self, api_keys: List[Tuple[str, str]], remove_bad_api_keys=True) -> None:
        self.remove_bad_api_keys = remove_bad_api_keys
        self.api_keys = Shifter(api_keys)

    def __call__(self, func: Callable):
        def res_func(*args, **kwargs):
            items_to_delete = []
            found = False
            for _ in range(len(self.api_keys)):
                i, api_key = self.api_keys()
                if i == -1:
                    break
                success = False
                for try_cnt in range(6):
                    todo = 'SUCCESS'

                    kwargs2 = kwargs.copy()
                    kwargs2['api_key'] = api_key

                    try:
                        res = func(*args, **kwargs2)
                    except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.UnprocessableEntityError) as e:
                        todo = 'RETRY'
                        if try_cnt >= 5:
                            raise Exception(e, args, kwargs)
                    except openai.AuthenticationError as e:
                        todo = 'DELETE_AND_TRY_NEXT'
                        print(api_key, 'is expired.')
                        items_to_delete.append(api_key)
                    except (openai.PermissionDeniedError, openai.RateLimitError) as e:
                        todo = 'TRY_NEXT'
                    except Exception as e:
                        raise Exception(e, args, kwargs)
                    
                    if todo == 'SUCCESS':
                        success = True
                        found = True
                        print(api_key, ' success')
                        break
                    if todo in ['DELETE_AND_TRY_NEXT', 'TRY_NEXT']:
                        break
                if success:
                    break
            
            if self.remove_bad_api_keys:
                self.api_keys.remove(items_to_delete)
            
            if found:
                return res
            else:
                raise NoValidAPIKey()
        
        return res_func
                    