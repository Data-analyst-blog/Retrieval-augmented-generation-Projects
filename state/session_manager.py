class SessionManager:

    def __init__(self):
        self.sessions = {}

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def update_history(self, session_id, user_msg, bot_msg):

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "user": user_msg,
            "bot": bot_msg
        })