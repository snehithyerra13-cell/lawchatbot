"""Auth0 OAuth2 / OpenID Connect Integration Service for ConstitutionGPT.

Handles Auth0 Universal Login URL generation, authorization code exchange,
user profile retrieval, session normalization, and guest/demo fallback.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Any, Dict, Optional

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Auth0Service:
    """Manages Auth0 authentication flows and user session decoding."""

    def __init__(
        self,
        domain: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        audience: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        # Always re-read from environment variables to pick up .env changes
        # without needing a server restart
        def _env_or_setting(env_key: str, setting_val: str) -> str:
            env_val = os.getenv(env_key, "").strip()
            return env_val if env_val else (setting_val or "").strip()

        self.domain = (domain if domain is not None else _env_or_setting("AUTH0_DOMAIN", settings.auth0_domain))
        self.domain = self.domain.replace("https://", "").replace("http://", "").strip("/")
        self.client_id = (client_id if client_id is not None else _env_or_setting("AUTH0_CLIENT_ID", settings.auth0_client_id))
        self.client_secret = (client_secret if client_secret is not None else _env_or_setting("AUTH0_CLIENT_SECRET", settings.auth0_client_secret))
        self.callback_url = (callback_url if callback_url is not None else _env_or_setting("AUTH0_CALLBACK_URL", settings.auth0_callback_url)) or "http://localhost:8501"
        self.audience = (audience if audience is not None else _env_or_setting("AUTH0_AUDIENCE", settings.auth0_audience))

    @property
    def is_configured(self) -> bool:
        """Return True if required Auth0 configuration credentials are present."""
        return bool(self.domain and self.client_id)

    def get_authorize_url(self, state: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """Generate Auth0 Universal Login URL for web redirect.
        
        Args:
            state: Optional CSRF token or session state identifier.
            prompt: Optional prompt parameter ('login', 'consent', 'none').
                    Omitted by default to avoid cookie transaction drops in privacy browsers.
            
        Returns:
            The complete URL string to redirect user for Auth0 authentication.
        """
        if not self.is_configured:
            return ""

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.callback_url,
            "scope": "openid profile email",
        }
        if prompt:
            params["prompt"] = prompt
        if self.audience:
            params["audience"] = self.audience
        if state:
            params["state"] = state

        query_str = urllib.parse.urlencode(params)
        return f"https://{self.domain}/authorize?{query_str}"

    def exchange_code_for_user(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for user access token and profile.
        
        Args:
            code: The authorization code returned by Auth0 via callback query parameter.
            
        Returns:
            Dictionary containing normalized user profile info or error details.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Auth0 domain or client_id is not configured.",
            }

        token_url = f"https://{self.domain}/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.callback_url,
        }

        try:
            resp = requests.post(token_url, data=payload, headers=headers, timeout=10)
            token_data = resp.json()

            if resp.status_code != 200 or "access_token" not in token_data:
                err_msg = token_data.get("error_description") or token_data.get("error") or resp.text
                logger.error("Auth0 token exchange failed (%s): %s", resp.status_code, err_msg)
                return {"success": False, "error": f"Token exchange failed: {err_msg}"}

            access_token = token_data.get("access_token")
            id_token = token_data.get("id_token")

            # Query Auth0 userinfo endpoint using access token
            user_info = self.fetch_user_profile(access_token)
            if not user_info:
                # Fallback to decoding unverified payload from id_token if userinfo is unavailable
                user_info = self._decode_id_token_payload(id_token)

            normalized_user = self.normalize_user_profile(user_info)
            normalized_user["access_token"] = access_token
            return {
                "success": True,
                "user": normalized_user,
            }

        except Exception as e:
            logger.exception("Exception during Auth0 token exchange: %s", e)
            return {"success": False, "error": str(e)}

    def fetch_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Fetch user profile from Auth0 /userinfo endpoint."""
        userinfo_url = f"https://{self.domain}/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            resp = requests.get(userinfo_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Failed to query Auth0 userinfo: %s", e)
        return {}

    def _decode_id_token_payload(self, id_token: Optional[str]) -> Dict[str, Any]:
        """Safely decode payload of JWT id_token without external network call."""
        if not id_token:
            return {}
        try:
            import jwt

            # Decode payload without verifying signature for instant local display
            return jwt.decode(id_token, options={"verify_signature": False})
        except Exception as e:
            logger.warning("Could not decode id_token: %s", e)
            return {}

    @staticmethod
    def normalize_user_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Auth0 / OpenID Connect profile dictionary."""
        sub = profile_data.get("sub", "")
        email = profile_data.get("email", "")
        name = (
            profile_data.get("name")
            or profile_data.get("nickname")
            or (email.split("@")[0] if email else "Advocate User")
        )
        picture = profile_data.get("picture", "")
        raw_uid = sub or f"user_{name.lower().replace(' ', '_')}"
        safe_uid = raw_uid.replace(".", "_")

        return {
            "user_id": safe_uid,
            "name": name,
            "email": email,
            "picture": picture,
            "nickname": profile_data.get("nickname", name),
            "is_authenticated": True,
            "auth_provider": "auth0",
            "raw": profile_data,
        }

    def get_logout_url(self, return_to: Optional[str] = None) -> str:
        """Construct Auth0 logout URL to clear Auth0 SSO session."""
        if not self.is_configured:
            return ""
        target = return_to or self.callback_url
        params = {
            "client_id": self.client_id,
            "returnTo": target,
        }
        return f"https://{self.domain}/v2/logout?{urllib.parse.urlencode(params)}"

    @staticmethod
    def get_demo_user(persona: str = "advocate") -> Dict[str, Any]:
        """Generate a simulated authenticated user profile for immediate offline testing."""
        personas = {
            "advocate": {
                "user_id": "auth0|demo_advocate_priya",
                "name": "Advocate Priya Sharma",
                "email": "priya.sharma@delhi-bar.org",
                "picture": "https://api.dicebear.com/7.x/initials/svg?seed=Priya+Sharma&backgroundColor=003366&textColor=ffffff",
                "nickname": "Priya",
                "is_authenticated": True,
                "auth_provider": "demo",
            },
            "scholar": {
                "user_id": "auth0|demo_scholar_vikram",
                "name": "Prof. Vikram Sen",
                "email": "v.sen@nludelhi.edu.in",
                "picture": "https://api.dicebear.com/7.x/initials/svg?seed=Vikram+Sen&backgroundColor=4b0082&textColor=ffffff",
                "nickname": "Vikram",
                "is_authenticated": True,
                "auth_provider": "demo",
            },
        }
        return personas.get(persona, personas["advocate"])

    @staticmethod
    def create_local_user(identifier: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a local user profile for username / email without external OAuth."""
        import re
        raw_clean = (identifier or "").strip()
        if not raw_clean:
            raw_clean = "user_guest"
        safe_uid = re.sub(r"[^a-zA-Z0-9_@-]", "_", raw_clean).lower().replace(".", "_")
        if not safe_uid.startswith("user_") and "@" not in safe_uid:
            user_id = f"user_{safe_uid}"
        else:
            user_id = safe_uid

        name = display_name or raw_clean.split("@")[0].replace("_", " ").title()
        return {
            "user_id": user_id,
            "name": name,
            "email": raw_clean if "@" in raw_clean else f"{safe_uid}@local.internal",
            "picture": f"https://api.dicebear.com/7.x/initials/svg?seed={name}&backgroundColor=1a73e8&textColor=ffffff",
            "nickname": name,
            "is_authenticated": True,
            "auth_provider": "local",
            "raw": {"id": user_id, "name": name},
        }

    @staticmethod
    def create_guest_user(guest_id: Optional[str] = None) -> Dict[str, Any]:
        """Create an isolated unique guest user profile so guest sessions don't collide."""
        import uuid
        gid = guest_id or f"guest_{uuid.uuid4().hex[:8]}"
        short_num = gid.split("_")[-1].upper()
        return {
            "user_id": gid,
            "name": f"Guest Advocate #{short_num}",
            "email": f"{gid}@guest.internal",
            "picture": f"https://api.dicebear.com/7.x/initials/svg?seed={gid}&backgroundColor=5f6368&textColor=ffffff",
            "nickname": f"Guest #{short_num}",
            "is_authenticated": False,
            "auth_provider": "guest",
            "raw": {"id": gid},
        }


_auth0_instance: Optional[Auth0Service] = None


def get_auth0_service() -> Auth0Service:
    """Get Auth0Service — always reads fresh from env so .env changes apply
    without a server restart."""
    # Re-read env vars every call so credentials saved to .env mid-session work immediately.
    import os
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parent.parent.parent / ".env"
        if _env.exists():
            load_dotenv(_env, override=True)
    except ImportError:
        pass
    return Auth0Service()
